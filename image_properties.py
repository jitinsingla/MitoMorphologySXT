'''
Wrapper class to handle data and retrieve it properties
'''

import json, os, sys, gc, csv, math
sys.path.append("../")
from pathlib import Path
import numpy as np
from read_write_mrc import read_mrc, write_mrc
from skimage import io
import matplotlib.pyplot as plt
# from A_codes.utils import *
from scipy.spatial.distance import euclidean
from scipy.ndimage.measurements import label as findIndividualLabel
from skimage.segmentation import find_boundaries
from scipy.ndimage.measurements import sum as SNMS
from scipy.ndimage.measurements import center_of_mass
from scipy import interpolate
from copy import deepcopy



### Because I am using MRC formats so I hard coded mrc to retreive the keys, labels and original lac value too ###
class ImageData():
    def __init__(self, jsonfile):
        self.path = Path(jsonfile).parent
        with open(jsonfile) as f:
            self.data = json.load(f)
        

        self._lac = None
        self._normalized_lac = None
        self._labelfield = None
        self._cell_mask = None
        self._nucleus_mask = None
        self._cytoplasm = None
        self._segmented_organelles = None
        self._int_key = None  # {organelle:int}
        self._int_key_reverse = None # {int:organelle}
        self._lac_factor = None
        self._labelLAC_values = None
        self._individual_labelled_organelles = {}
    
    @property
    def int_key(self):
        if self._int_key is None:
            self._int_key = {k: v for k,v in self.data['mrc']['key'].items()}
        return self._int_key

    @property
    def int_key_reverse(self):
        if self._int_key_reverse is None:
            self._int_key_reverse = {v: k for k, v in self.data['mrc']['key'].items()}
        return self._int_key_reverse

    @property
    def segmented_organelles(self):
        ''' It lists all the segmented organelles available in particular cell '''
        '''
        if self._segmented_organelles == None:
            labels = [int(_) for _ in np.unique(self.labelfield)]
            self._segmented_organelles = [self._int_key_reverse[int(_)] for _ in labels]
        return self._segmented_organelles
        '''
        return self.int_key.keys()

    @property
    def lac_factor(self):
        if self._lac_factor is None:
            self._lac_factor = self.data['lac_factor']
        return self._lac_factor
 
         
    @property
    def lac(self):
        ''' loads LAC '''
        '''Here we don't have original mrc in same folder. So I can use tiff file'''
        if self._lac is None:
            self._lac = read_mrc(str(self.path / self.data['mrc']['lac']))
            self._lac *= self.lac_factor
        return self._lac

    '''Keep it for later use in case'''
    @property
    def normalized_lac(self):
        ''' Lazy loading normalized_lac '''
        if self._normalized_lac is None:
            mean_val = np.mean(self.lac[self.cell_mask(tag) > 0])
            std_val = np.std(self.lac[self.cell_mask(tag) > 0])
            logging.info('Normalizing with mean {:.3} std: {:.3}'.format(
                mean_val, std_val))

            self._normalized_lac = (self.lac - mean_val) / std_val
            bg_val = np.percentile(self._normalized_lac[self.cell_mask(tag) > 0], 1)
            self._normalized_lac[self.cell_mask(tag) == 0] = bg_val
        return self._normalized_lac

    # @property
    def labelLACvalues(self, label, tag):
        
        mrcfile = read_mrc(str(self.path/self.data['mrc']['lac']))
        mask = self.label(label, tag)
        mask_2_copy = deepcopy(mask)
        inds_mask = np.where(mask_2_copy == 1)
      
        Z = list(inds_mask[0])
        Y = list(inds_mask[1])
        X = list(inds_mask[2])
        assert len(X) == len(Y) == len(Z), "Sanity check failed: Unequal lengths of X, Y, Z"
        assert mrcfile.shape == mask.shape, f"Shape mismatch: Raw Shape {mrcfile.shape} , Mask Shape {mask.shape}"
        tmp_lac = [self.lac[Z[i], Y[i], X[i]] for i in range(len(X))] #*self.data['lac_factor'] 
        
        return tmp_lac
    
    def labelfield(self, tag):
        if self._labelfield is None:
            if tag == 'groundtruth':
                self._labelfield = read_mrc(self.path / self.data['mrc']['labelfield'])
            elif tag == 'prediction':
                self._labelfield = read_mrc(self.path / self.data['mrc']['prediction_mask'])
        return self._labelfield

    # @property
    def cell_mask(self, tag):
        ''' load cell_mask '''
        if self._cell_mask is None:
            cell_labels = [v for k, v in self.int_key.items() if k!="exterior"]
            self._cell_mask = np.isin(self.labelfield(tag), cell_labels).astype(int)
        return self._cell_mask

    # @property
    def cytoplasm(self, tag):
        ''' load cytoplasm '''
        if self._cytoplasm is None:
            self._cytoplasm = (self.cell_mask(tag)==1)*(self.nucleus_mask(tag)==0)    # *(self.label('mitochondria')==0)*(self.label('granule')==0) ## cytoplasm incudes both mito and vesicles
        return self._cytoplasm
    
    # @property
    def nucleus_mask(self, tag):
        ''' load cell_mask '''
        if self._nucleus_mask is None:
            #cell_labels = [v for k, v in self.int_key.items() if k!="exterior"]
            self._nucleus_mask = np.isin(self.labelfield(tag), [2,7]).astype(int) # Nucleus and Nucleoli
        return self._nucleus_mask
    
    def labelVolumevalues(self, label, tag):
        # if key not in self.int_key.keys():
        #     return None
        labelMask = self.label(label, tag)

        if labelMask is not None:
            volumeVoxels = labelMask.sum()
            volume_um = volumeVoxels*((1 / self.lac_factor)**3)
            return volume_um, volumeVoxels
        else:
            return None

    def labelVolumevaluesIndividual(self, label, tag):
        VolumeList_um = []
        VolumeVoxelList = []
        CoordinatesList = []
        labelMask = self.label(label, tag)
        if labelMask is not None:
            if label in self._individual_labelled_organelles:
                organelle_labelled_arr = self._individual_labelled_organelles[label]
                numIndividualorganelle = organelle_labelled_arr.max()
            else:
                structure = np.ones((3, 3, 3), dtype=np.int32)
                organelle_labelled_arr, numIndividualorganelle = findIndividualLabel(labelMask, structure)
                self._individual_labelled_organelles[label] = organelle_labelled_arr
            VolumeVoxelList = np.array([(organelle_labelled_arr==i).sum() for i in range(1, numIndividualorganelle+1)])
            CoordinatesList = np.array([[coords[0][0], coords[1][0], coords[2][0]] for i in range(1, numIndividualorganelle + 1) for coords in [np.where(organelle_labelled_arr == i)]])

            VolumeList_um = VolumeVoxelList*((1 / self.lac_factor)**3)
            return VolumeList_um, VolumeVoxelList, numIndividualorganelle, CoordinatesList
        else:
            return None
# calculate wholecell and individual SA
    def labelSAvaluesIndividual(self, label, tag):
        def calculate_surface_area_3d(data):
            indices = np.where(data!=0)
            surface_area =  np.array([calculate_voxel_area(data, indices[1][i], indices[0][i], indices[2][i])           for i in range(len(indices[0]))]).sum()
            return surface_area

        def calculate_voxel_area(data, x, y, z):
            # Calculate the surface area of a voxel
            area = 0
            ny, nx, nz = data.shape

            # Check each face of the voxel if it is exposed to the surface
            if x == 0 or data[y, x-1, z] == 0:  # Left face
                area += 1
            if x == nx - 1 or data[y, x+1, z] == 0:  # Right face
                area += 1
            if y == 0 or data[y-1, x, z] == 0:  # Bottom face
                area += 1
            if y == ny - 1 or data[y+1, x, z] == 0:  # Top face
                area += 1
            if z == 0 or data[y, x, z-1] == 0:  # Front face
                area += 1
            if z == nz - 1 or data[y, x, z+1] == 0:  # Back face
                area += 1
            return area
        
        SurfaceA_um = []
        SurfaceArea = []
        labelMask = self.label(label,tag)
        if labelMask is not None:
            if label in self._individual_labelled_organelles:
                organelle_labelled_arr = self._individual_labelled_organelles[label]
                numIndividualorganelle = organelle_labelled_arr.max()
            else:
                structure = np.ones((3, 3, 3), dtype=np.int32)
                organelle_labelled_arr, numIndividualorganelle = findIndividualLabel(labelMask, structure)
                self._individual_labelled_organelles[label] = organelle_labelled_arr
           
            for j in range(1, numIndividualorganelle+1):
                # surface_area_old = calculate_surface_area_3d_old(np.array(organelle_labelled_arr==j, dtype=int))
                surface_area = calculate_surface_area_3d(np.array(organelle_labelled_arr==j, dtype=int))
                SurfaceArea.append(surface_area)
                SurfaceA_um.append(surface_area*((1 / self.lac_factor)**2))
                del surface_area
            del organelle_labelled_arr
       
        return SurfaceA_um, np.sum(SurfaceA_um), SurfaceArea, np.sum(SurfaceArea), numIndividualorganelle
    
    def label(self, key,tag):
        '''
        Arguments:
            key {string} -- dictionary key
        Returns:
            nparray -- Image corresponding the the chosen label
        '''
        if key=='cytoplasm':  return self.cytoplasm(tag)
        elif key not in self.int_key.keys():  return None
        elif key=='membrane': return self.cell_mask(tag)
        elif key=='nucleus':    return self.nucleus_mask(tag)
        else:   return (self.labelfield(tag) == self.int_key[key]).astype(int)
