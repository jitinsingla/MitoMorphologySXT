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
from scipy.spatial.distance import euclidean
from scipy.ndimage.measurements import label as findIndividualLabel
from skimage.segmentation import find_boundaries
from copy import deepcopy


class ImageData():
    """
    Wrapper class for handling image data and extracting organelle properties.

    Parameters
    ----------
    jsonfile : str
        Path to metadata JSON file.
    """
    
    def __init__(self, jsonfile):
        self.path = Path(jsonfile).parent
        with open(jsonfile) as f:
            self.data = json.load(f)
        
        # Lazy loaded variables
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
        """Dictionary mapping organelle → integer label"""
        if self._int_key is None:
            self._int_key = {k: v for k,v in self.data['mrc']['key'].items()}
        return self._int_key

    @property
    def int_key_reverse(self):
        if self._int_key_reverse is None:
            self._int_key_reverse = {v: k for k, v in self.data['mrc']['key'].items()}
        return self._int_key_reverse

    @property
    def lac_factor(self):
        """Return LAC scaling factor"""
        if self._lac_factor is None:
            self._lac_factor = self.data['lac_factor']
        return self._lac_factor
 
         
    @property
    def lac(self):
        ''' Loads LAC '''
        '''Return LAC Intensity * LAC factor'''
        if self._lac is None:
            self._lac = read_mrc(str(self.path / self.data['mrc']['lac']))
            self._lac *= self.lac_factor
        return self._lac


    @property
    def labelLACvalues(self, label):
        """
        Extract LAC values of voxels belonging to a given organelle label.
    
        Args:
            label (str): organelle name used to generate the binary mask
                         (converted internally to a numpy array)
        """
        mrcfile = read_mrc(str(self.path/self.data['mrc']['lac']))
        mask = self.label(label)
        mask_2_copy = deepcopy(mask)
        inds_mask = np.where(mask_2_copy == 1)
      
        Z = list(inds_mask[0])
        Y = list(inds_mask[1])
        X = list(inds_mask[2])
        assert len(X) == len(Y) == len(Z), "Sanity check failed: Unequal lengths of X, Y, Z"
        assert mrcfile.shape == mask.shape, f"Shape mismatch: Raw Shape {mrcfile.shape} , Mask Shape {mask.shape}"
        tmp_lac = [self.lac[Z[i], Y[i], X[i]] for i in range(len(X))] 
        
        return tmp_lac
        
    @property
    def labelfield(self):
        ''' loads labelfield '''
        if self._labelfield is None:
            self._labelfield = read_mrc(self.path / self.data['mrc']['labelfield']) 
        return self._labelfield    

    @property
    def cell_mask(self):
        ''' load cell_mask '''
        if self._cell_mask is None:
            cell_labels = [v for k, v in self.int_key.items() if k!="exterior"]
            self._cell_mask = np.isin(self.labelfield, cell_labels).astype(int)
        return self._cell_mask

    @property
    def cytoplasm(self):
        ''' load cytoplasm.
            Cytoplasm includes mitochondria and vesicles.
        '''
        if self._cytoplasm is None:
            self._cytoplasm = (self.cell_mask ==1)*(self.nucleus_mask ==0)   
        return self._cytoplasm
    
    @property
    def nucleus_mask(self):
        ''' load cell_mask '''
        if self._nucleus_mask is None:
            self._nucleus_mask = np.isin(self.labelfield,[2,7]).astype(int)
        return self._nucleus_mask
    
    def labelVolumevalues(self, label):
        """
        Calculate total volume of a given organelle label.
    
        Args:
            label (str): organelle name used to generate the binary mask
                                 (converted internally to a numpy array)
        """
        labelMask = self.label(label)

        if labelMask is not None:
            volumeVoxels = labelMask.sum()
            volume_um = volumeVoxels*((1 / self.lac_factor)**3)
            return volume_um, volumeVoxels
        else:
            return None

    def labelVolumevaluesIndividual(self, label):
        """
        Calculate volumes of individual organelles for a given label.
    
        Args:
            label (str): organelle name used to generate the binary mask
                                 (converted internally to a numpy array)
        """
        VolumeList_um = []
        VolumeVoxelList = []
        CoordinatesList = []
        labelMask = self.label(label)
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

            
    def labelSAvaluesIndividual(self, label):
        """
        Calculate surface area of individual organelles for a given label.
    
        Args:
            label (str): organelle name used to generate the binary mask
                                 (converted internally to a numpy array)
    
        """
        def calculate_surface_area_3d(data):
            # compute exposed voxel faces
            indices = np.where(data!=0)
            surface_area =  np.array([calculate_voxel_area(data, indices[1][i], indices[0][i], indices[2][i])           for i in range(len(indices[0]))]).sum()
            return surface_area

        def calculate_voxel_area(data, x, y, z):
        # compute exposed faces of a single voxel
            area = 0
            ny, nx, nz = data.shape
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
        labelMask = self.label(label)
        if labelMask is not None:
            if label in self._individual_labelled_organelles:
                organelle_labelled_arr = self._individual_labelled_organelles[label]
                numIndividualorganelle = organelle_labelled_arr.max()
            else:
                structure = np.ones((3, 3, 3), dtype=np.int32)
                organelle_labelled_arr, numIndividualorganelle = findIndividualLabel(labelMask, structure)
                self._individual_labelled_organelles[label] = organelle_labelled_arr
           
            for j in range(1, numIndividualorganelle+1):
                surface_area = calculate_surface_area_3d(np.array(organelle_labelled_arr==j, dtype=int))
                SurfaceArea.append(surface_area)
                SurfaceA_um.append(surface_area*((1 / self.lac_factor)**2))
                del surface_area
            del organelle_labelled_arr
       
        return SurfaceA_um, np.sum(SurfaceA_um), SurfaceArea, np.sum(SurfaceArea), numIndividualorganelle
    
    def label(self, key):
        '''
        Arguments:
            key {string} -- dictionary key

        '''
        if key=='cytoplasm':  return self.cytoplasm
        elif key not in self.int_key.keys():  return None
        elif key=='membrane': return self.cell_mask
        elif key=='nucleus':    return self.nucleus_mask
        else:   return (self.labelfield == self.int_key[key]).astype(int)
