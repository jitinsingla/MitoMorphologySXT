import json, os, csv, gc, copy,math, glob
import datetime
import numpy as np
import pandas as pd
import pickle, time
import path_settings
import seaborn as sns
import multiprocessing
from skimage import io
import SimpleITK as sitk
from copy import deepcopy
from image_properties import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from functools import partial
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from scipy.spatial.distance import euclidean
from read_write_mrc import read_mrc, write_mrc
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.ticker import FormatStrFormatter



def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
Dict2 = { '0mM' : ['783_5_pre_rec', '784_5_pre_rec','783_6_pre_rec','783_12_pre_rec'],
         '25mM_Gln_5min' : ['1060_4-6_pre_rec', '1092_11_pre_rec','1044_7_pre_rec','1044_8-9_pre_rec']
         }
            
def find(path,tag=None):
    '''Finds all the samples by searching for .json files
    Arguments:
        path {[type]} -- root directory for the walk
        tag {[type]} -- if defined, checks that the path contains the tag (e.g. for Processed vs Unprocessedc data)
    
    Returns:
        [type] -- List of strings containing the path to the json files 
    '''
    jsonfilelist = []
    predictionMRCfilelist = []
    LabelMRCfilelist = []
    RawMRCfilelist = []

    for dirpath, _, files in os.walk(path, followlinks=True):
        for x in files:
            if x.endswith('.json') and tag in dirpath:
                jsonfilelist.append(os.path.join(dirpath, x))
            if x.endswith("pre_rec_labels.mrc") and tag in dirpath:
                LabelMRCfilelist.append(os.path.join(dirpath, x))
            if x.endswith("pre_rec.mrc") and tag in dirpath:
                RawMRCfilelist.append(os.path.join(dirpath, x))
    return jsonfilelist, predictionMRCfilelist,LabelMRCfilelist,RawMRCfilelist

def sortedFileList():
    
    jsonFileList = []
    predictionMRCfilelist = []
    LabelMRCfilelist = []
    RawMRCfilelist = []
    
    for k, v in Dict2.items():
        
        filelist_item1 = [find(path_settings.COLLECTION, str(v[key]))[0][0] for key in range(len(v)) ]
        jsonFileList.extend(filelist_item1)
        filelist_item3 = [find(path_settings.COLLECTION, str(v[key]))[2][0] for key in range(len(v))]
        LabelMRCfilelist.extend(filelist_item3)
        filelist_item4 = [find(path_settings.COLLECTION, str(v[key]))[3][0] for key in range(len(v))]
        RawMRCfilelist.extend(filelist_item4)
    
    return jsonFileList, LabelMRCfilelist, RawMRCfilelist
            

################################### Extract Label LAC Values: Parallel code ###################################
def labelLACvaluesLocal(args, label, conditions, collectCompleteList):
    i, jsonfile = args
    cell = ImageData(jsonfile)
    lacValues = cell.labelLACvalues(label)
    meanLacValue = np.mean(lacValues)
    if collectCompleteList:        
        resultDict = {'i':i,
                'cell_id': conditions[cell.data["name"]]['cell'],
                'conditions': conditions[cell.data["name"]]['condition'],
                'voxel_LAC': lacValues,
                'MeanVoxelLAC': meanLacValue}
    else:
        del lacValues
        gc.collect()
        resultDict = {'i':i,
                'cell_id': conditions[cell.data["name"]]['cell'],
                'conditions': conditions[cell.data["name"]]['condition'],
                'MeanVoxelLAC': meanLacValue}
    return resultDict

def labelLACvaluesParallel(label, filelist, conditions, collectCompleteList=False, njobs=None):
    """
    Parallel wrapper to compute LAC statistics for multiple files.

    Parameters
    ----------
    label : str
        Label identifier used to extract LAC values.

    filelist : list
        List of JSON file paths to process.

    conditions : dict
        Dictionary containing cell metadata and experimental conditions.

    collectCompleteList : bool, optional (default=False)
        If True, stores full voxel LAC distributions.
        If False, stores only mean LAC values (recommended for large datasets).

    njobs : int, optional (default=None)
        Number of CPU cores to use.
        If None, uses all available CPU cores.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing LAC statistics and metadata for all cells.
    """
    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    print("---------------------------------------------------------")
    dictLac = {}
    if njobs is None:
        njobs = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=njobs) as pool:
        partialFunction = partial(labelLACvaluesLocal, label=label, conditions=conditions, collectCompleteList=collectCompleteList)
        results = pool.map(partialFunction, enumerate(filelist))

    for result in results:
        i = result['i']
        del result['i']
        dictLac[i] = result
    del results
    gc.collect()        
    
    pool.close()
    pool.terminate()
    # convert to dataframe
    dfLac = pd.DataFrame(dictLac).transpose()
    del dictLac
    gc.collect()
    
    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    return dfLac

################################### Extract Label Volume Values: Parallel code ###################################
def labelVolumeSAvaluesLocal(args, label, conditions, volSAFlag, collectIndividualList):
    """
    Process a single JSON file and extract organelle volume and/or surface area.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        i : int
            Index of the file in the input list.
        jsonfile : str
            Path to the JSON file.

    label : str
        Organelle label name (example: mitochondria, nucleus, cytoplasm).

    conditions : dict
        Dictionary containing cell metadata.
        Expected structure:
        conditions[cell_name]['cell'] → cell ID
        conditions[cell_name]['condition'] → experimental condition

    volSAFlag : str
        Controls which measurements to extract:
        "vol"  → volume only
        "sa"   → surface area only
        "both" → both volume and surface area

    collectIndividualList : bool
        If True, stores measurements of individual organelles.
        If False, stores only total measurements.

    Returns
    -------
    dict or tuple(dict, dict)
        resultDict :
            Dictionary containing total measurements and metadata.

        resultDictIndi :
            Dictionary containing individual organelle measurements
            (only returned if collectIndividualList=True and applicable).
    """
    
    i, jsonfile = args
    cell = ImageData(jsonfile)
    resultDict = {'i':i}
    resultDict['cell_id'] = conditions[cell.data["name"]]['cell']
    resultDict['conditions'] = conditions[cell.data["name"]]['condition']
    if volSAFlag=="sa":
        collectCompleteList = True
    if collectIndividualList:
        resultDictIndi = deepcopy(resultDict)
        if volSAFlag=="vol" or volSAFlag=="both":
            resultDictIndi['volume_um'], resultDictIndi['volume_voxels'], resultDict['numIndividualorganelleVol'], _ = cell.labelVolumevaluesIndividual(label)
        if volSAFlag=="sa" or volSAFlag=="both":
            resultDictIndi["SurfaceA_um"], resultDict["TotalSurfaceA_um"], resultDictIndi["SurfaceArea"], resultDict["TotalSurfaceArea"], resultDict["numIndividualorganelleSA"] = cell.labelSAvaluesIndividual(label)
    
    if volSAFlag=="vol" or volSAFlag=="both":
        resultDict['TotalVolume_um'], resultDict['TotalVolume_voxels'] = cell.labelVolumevalues(label)
    if label=="nucleus" or label=="cytoplasm" or label=="membrane":
        return resultDict
    return resultDict, resultDictIndi

def labelVolumeSAvaluesParallel(label, filelist, conditions, volSAFlag="both", collectIndividualList=False, njobs=None):
    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    print("---------------------------------------------------------")
    volSADict = {}
    volSADictIndi = {}
    if njobs is None:
        njobs = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=njobs) as pool:
        partialFunction = partial(labelVolumeSAvaluesLocal, label=label, conditions=conditions, volSAFlag=volSAFlag, collectIndividualList=collectIndividualList)
        results = pool.map(partialFunction, enumerate(filelist))
    if label=="nucleus" or label=="cytoplasm" or label=="membrane":
        for result in results:
            i = result['i']
            del result['i']
            volSADict[i] = result
            
        del results
        gc.collect()        

        pool.close()
        pool.terminate()
        # convert to dataframe
        dfVolSA = pd.DataFrame(volSADict).transpose()
        del volSADict
        gc.collect()
        print("---------------------------------------------------------")
        print(datetime.datetime.now())
        return dfVolSA
    else:
        for (result, resultIndi) in results:
            i = result['i']
            del result['i'], resultIndi['i']
            volSADict[i] = result
            volSADictIndi[i] = resultIndi

        del results
        gc.collect()        

        pool.close()
        pool.terminate()
        # convert to dataframe
        dfVolSA = pd.DataFrame(volSADict).transpose()
        dfVolSAIndi = pd.DataFrame(volSADictIndi).transpose()
        del volSADict, volSADictIndi
        gc.collect()
        df_explode = dfVolSAIndi.explode(['SurfaceA_um', 'volume_um'])
        print("---------------------------------------------------------")
        print(datetime.datetime.now())
        return dfVolSA, dfVolSAIndi, df_explode

##################
def LacMeanLocal(args, label,conditions):
    """
    Extract mean LAC and organelle statistics for a single cell.

    Args:
        args (tuple): (index, json file path)
        label (str): organelle label name
        conditions (dict): cell metadata dictionary

    Returns:
        dict: mean LAC, organelle count, coordinates and volume
    """
    i, jsonfile = args
    cell = ImageData(jsonfile)
    Meanlac, numIndividualorganelle, OrganelleCenterOfMass, OrganelleVol = cell.LAC_mean(label)
    results = {'i':i,
            'cell': conditions[cell.data["name"]]['cell'],
            'conditions': conditions[cell.data["name"]]['condition'],
            'MeanLAC': Meanlac,
            'nomito': numIndividualorganelle,
            'Coordinates(ZYX)': OrganelleCenterOfMass,
            'ves_vol': OrganelleVol}
    
    return results
    
def LACMeanParallel(label, filelist, conditions, njobs=None):
    """
    Parallel computation of mean LAC values for multiple cells.

    Args:
        label (str): organelle label name
        filelist (list): list of json files
        conditions (dict): cell metadata
        njobs (int): number of parallel processes

    Returns:
        DataFrame: mean LAC statistics for all cells
    """
    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    print("---------------------------------------------------------")
    dictLac = {}
    if njobs is None:
        njobs = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=njobs) as pool:
        partialFunction = partial(LacMeanLocal, label=label, conditions=conditions)
        results = pool.map(partialFunction, enumerate(filelist))

    for result in results:
        i = result['i']
        del result['i']
        dictLac[i] = result
    del results
    gc.collect()        
    
    pool.close()
    pool.terminate()
    # convert to dataframe
    dfLac = pd.DataFrame(dictLac).transpose()
    del dictLac
    gc.collect()
    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    print("-------------------------SurfaceA_um--------------------------------")
    return dfLac

################################################################################################################
# Mitochondrial complexity Index
def calculateMCILocal(args, label, conditions):
    """
    Calculate Mitochondrial Complexity Index (MCI) for a single cell.

    Args:
        args (tuple): (index, json file path)
        label (str): organelle label name
        conditions (dict): cell metadata dictionary

    Returns:
        dict: individual and whole-cell MCI values
    """
    i, jsonfile = args
    cell = ImageData(jsonfile)
    volume_um, _, _,_ = cell.labelVolumevaluesIndividual(label)
    SurfaceA_um, _, SurfaceArea, _, _ = cell.labelSAvaluesIndividual(label)
    pi = math.pi
    b = 3/2
    individualMCI = (np.array(SurfaceA_um)**b) / (4 * pi * np.array(volume_um))
    WholeMCI = (np.sum(SurfaceA_um)**b) / (4 * pi * np.sum(volume_um))
    resultDict = {
        'i': i,
        'cell_id': conditions[cell.data["name"]]['cell'],
        'conditions': conditions[cell.data["name"]]['condition'],
        'individualMCI': list(individualMCI),
        'WholeMCI' : WholeMCI
    }
   
    return resultDict

def calculateMCIParallel(jsonFileList, label, conditions,Cell_IDs, njobs=None):
    """
    Parallel computation of MCI for multiple cells.

    Args:
        jsonFileList (list): list of json files
        label (str): organelle label name
        conditions (dict): cell metadata
        Cell_IDs (list): list of cell identifiers
        njobs (int): number of parallel processes

    Returns:
        DataFrame: MCI values for all cells
    """
    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    print("---------------------------------------------------------")
    
    if njobs is None:
        njobs = multiprocessing.cpu_count() - 10
    
    with multiprocessing.Pool(processes=njobs) as pool:
        partialFunction = partial(calculateMCILocal, label=label, conditions=conditions)
        results = pool.map(partialFunction, enumerate(jsonFileList))
    MCIdict = {}    
    for result in results:
        i = result['i']
        del result['i']
        MCIdict[i] = result
    del results
    
    MCIdf = pd.DataFrame(MCIdict).transpose()
    del MCIdict
    gc.collect()
    
    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    print("---------------------------------------------------------")
    
    return MCIdf
####################################################################################################

ContourDF = './Output/Contours_analysis/'
def RadialContour(jsonFileList, conditions, size = "global", df_save = False, save_edt = False):
    """
    Compute mitochondrial radial enrichment relative to nucleus–membrane axis.

    Method
    ------
    1. Generate cytosolic mask (cell – nucleus)
    2. Compute distance transform from nucleus and membrane
    3. Compute normalized radial fraction:
            fraction = d_nucleus / (d_nucleus + d_membrane)

    4. Divide cytosol into contour bins
    5. Compute expected voxel distribution
    6. Compute observed mitochondrial distribution
    7. Calculate enrichment:
            enrichment = observed / expected

    Parameters
    ----------
    jsonFileList : list
        List of cell metadata files

    conditions : dict
        Experimental condition mapping

    size : str
        Mito classification:
            global
            fragmented
            intermediate
            interconnected

    df_save : bool
        Save LAC dataframe

    save_edt : bool
        Save distance maps

    Returns
    -------
    dfLAC : DataFrame
        Per voxel LAC values

    all_enrichment_data : DataFrame
        Contour enrichment statistics
    """
    
    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    print("---------------------------------------------------------")
    all_enrichment_data = []
    LacData = []
    for i in range(len(jsonFileList)):
        cell = ImageData(jsonFileList[i])
        mitomask = cell.label('mitochondria')
        
        # 26-connected neighbourhood for 3D organelle connectivity
        # Ensures diagonally touching voxels belong to same mitochondrion     
        structure = np.ones((3, 3, 3), dtype=np.int32)# for 26 neighbour connectivity
        organelle_labelled_arr, numIndividualorganelle = findIndividualLabel(mitomask, structure)
        IndilabelMitoArray = []
        for j in range(1,numIndividualorganelle+1):
            IndimitoMask = np.where(organelle_labelled_arr==j)
            mitoVoxelCount = len(IndimitoMask[0])
            IndiVolume_um = mitoVoxelCount*((1 / cell.lac_factor)**3)
            # Remove segmentation noise (objects smaller than 10 voxels)
            # Threshold determined empirically from segmentation artifacts            
            if mitoVoxelCount >= 10:
                if size == "global":
                    IndilabelMitoArray.append(j)
            
                elif size == "fragmented" and 0 < IndiVolume_um < 0.06:
                    IndilabelMitoArray.append(j)
            
                elif size == "intermediate" and 0.06 <= IndiVolume_um <= 0.6:
                    IndilabelMitoArray.append(j)
            
                elif size == "interconnected" and IndiVolume_um > 0.6:
                    IndilabelMitoArray.append(j)
		 
        MitoArray = np.where(np.isin(organelle_labelled_arr, IndilabelMitoArray), 1, 0)
        sizeMitoArray = np.sum(MitoArray)

        cell_mask = cell.cell_mask
        nucleus_mask = cell.nucleus_mask
        cytosol_mask = cell_mask & (~nucleus_mask)
        d_nuc = distance_transform_edt(nucleus_mask==0)
        d_nuc[cytosol_mask == 0] = 0
        d_mem = distance_transform_edt(cell_mask)
        d_mem[cytosol_mask == 0] = 0
        # if save_edt:
        #     FolderPathNuc15 = './Output/Contours_analysis/CytoToNucDistance/'
        #     FolderPathCellMask15 = './Output/Contours_analysis/CellMaskToCellEdgeDistance/'
        #     os.makedirs(FolderPathNuc15,exist_ok=True)
        #     os.makedirs(FolderPathCellMask15,exist_ok=True)

        #     filenamenuc = FolderPathNuc15 + conditions[cell.data['name']]['cell']+'_edt'+'.mrc'
        #     write_mrc(filenamenuc, d_nuc.astype(np.float32))
        #     filenamecellMask = FolderPathCellMask15 + conditions[cell.data['name']]['cell']+'_edt'+'.mrc'
        #     write_mrc(filenamecellMask,d_mem.astype(np.float32))
        d_total = d_nuc + d_mem
        fraction = np.divide(
            d_nuc,
            d_total,
            out=np.zeros_like(d_nuc),
            where=d_total!=0
        )


        sliceIndex = cell_mask.shape[0]//2

        ncontours = 15
        contours = np.floor(fraction * ncontours).astype(np.int32)  # values from 0 to 9
        contours[cytosol_mask==0] = -1  # mask out background
        # plt.imshow(contours[sliceIndex], cmap='gray')
        #plt.savefig("testPILR.png", dpi=150)
        # plt.show()

        cytosol_mask_sum = cytosol_mask.sum()
        countourCount = np.asarray([(contours==nc).sum() for nc in range(ncontours)], dtype=np.float64)
        # Expected distribution assumes uniform random mitochondrial placement
        expectedDistribution = countourCount/cytosol_mask_sum
        mitoVol = mitomask.sum()

        mitoVoxelCount = []
        for c in range(ncontours):
            contourMask = contours==c
            mitoVoxelCount.append(MitoArray[contourMask].sum())
        mitoVoxelCount = np.asarray(mitoVoxelCount, dtype=np.float64)
        
        # Observed mitochondrial voxel distribution
        observedDistribution = mitoVoxelCount/mitoVol
        SizeMitoVolobservedDistribution= mitoVoxelCount/sizeMitoArray
        enrichment = observedDistribution/expectedDistribution
        size_enrichment = SizeMitoVolobservedDistribution/expectedDistribution
        # print('Observed Distribution and enrichment Done')
        # plt.figure(figsize=(5, 5))
        # plt.scatter(range(ncontours), enrichment, s=10)
        # plt.hlines(1, 0, ncontours-1, color="red")
        # plt.plot(enrichment)
        # plt.title(f"Enrichment plot of {conditions[cell.data['name']]['cell']} (Manual Segmentation)")
        # plt.savefig(ContourDF+'Enrichment_'+conditions[cell.data['name']]['cell']+'.png', dpi=150)
        # plt.show()

        # plt.figure(figsize=(5, 5))
        # plt.scatter(range(ncontours), observedDistribution, s=10)
        # # plt.hlines(1, 0, ncontours-1, color="red")
        # plt.plot(observedDistribution)
        # plt.title(f"Observed Mito Distribution of {conditions[cell.data['name']]['cell']} (Manual Segmentation)")
        # plt.savefig(ContourDF+'ObservedMitoDist_'+conditions[cell.data['name']]['cell']+'.png', dpi=150)
        # plt.show()
            # Collect enrichment data for selected conditions
        Cell_ID = conditions[cell.data['name']]['cell']
        if size != 'global':
            for contour_bin in range(ncontours):
                all_enrichment_data.append({
                    'CellID': Cell_ID,
                    'Condition':conditions[cell.data['name']]['condition'],
                    'ContourBin': contour_bin,
                    'ExpectedVoxelDistribution':expectedDistribution[contour_bin],
                    'ObservedMitoDistribution':observedDistribution[contour_bin],
                    'SizeMitoVolobservedDistribution':SizeMitoVolobservedDistribution[contour_bin],
                    'size_enrichment':size_enrichment[contour_bin],
                    'Enrichment': enrichment[contour_bin]
                })
        else:
            for contour_bin in range(ncontours):
                all_enrichment_data.append({
                    'CellID': Cell_ID,
                    'Condition':conditions[cell.data['name']]['condition'],
                    'ContourBin': contour_bin,
                    'ExpectedVoxelDistribution':expectedDistribution[contour_bin],
                    'ObservedMitoDistribution':observedDistribution[contour_bin],
                    'Enrichment': enrichment[contour_bin]
                })

        rawMrc = cell.lac
        Z, Y, X = np.where(MitoArray == 1)
        bins_indexs = contours[Z,Y,X]
        MitoDistanceRatios = fraction[Z,Y,X]
        cond = conditions[cell.data['name']]['condition']

        for num in range(len(Z)):
            bins_index = bins_indexs[num]
            Mito_ratio = MitoDistanceRatios[num]
            LAC_value = rawMrc[Z[num],Y[num],X[num]]
            cond = conditions[cell.data['name']]['condition']
            LacData.append([Cell_ID, cond, bins_index, LAC_value, Mito_ratio])

    all_enrichment_data = pd.DataFrame(all_enrichment_data)
    dfLAC = pd.DataFrame(LacData, columns=['CellID', 'Condition', 'BinIndex', 'MitoVoxelLAC', 'DistanceRatio'])
    all_enrichment_data.to_csv(os.path.join(ContourDF,"all_enrichment_data.csv"),index = False)
    if df_save:
        dfLAC.to_csv(os.path.join(ContourDF,"ContourLACdf.csv"),index = False)

    print("---------------------------------------------------------")
    print(datetime.datetime.now())
    print("---------------------------------------------------------")
    return dfLAC, all_enrichment_data

######################################################################################


def stats_boxplot(
    data,
    x,
    y,
    ylabel="",
    title="",
    alpha=0.05,
    palette=None,
    show_points=False,
    save_path=None
):

    """
    Automatically selects appropriate statistical test and generates boxplot.

    Statistical decision workflow:
    1. Test normality (Shapiro-Wilk)
    2. If normal → test equal variance (Levene)
    3. If normal + equal variance → ANOVA + Tukey HSD
    4. Otherwise → Kruskal-Wallis + Dunn test

    Parameters
    ----------
    data : pandas DataFrame
        Input dataframe containing measurements

    x : str
        Grouping column name

    y : str
        Measurement column name

    ylabel : str
        Label for Y axis

    title : str
        Plot title

    alpha : float
        Significance level (default 0.05)

    palette : dict or seaborn palette
        Colors for groups

    show_points : bool
        Show individual datapoints

    save_path : str
        File path to save figure

    Returns
    -------
    stats_table : DataFrame
        Posthoc comparison table

    test_used : str
        Name of statistical test used
    """

    # ---------- Set default colors ----------
    if palette is None:
        palette = sns.color_palette("Set2")


    # ---------- Extract groups for statistical testing ----------
    # Each condition becomes one array
    groups = [
        group[y].dropna().values
        for _,group in data.groupby(x)
    ]


    # ---------- Normality testing ----------
    # Shapiro test used because sample sizes are small
    normal=True

    normality_results=[]

    for g in groups:

        # Skip very small samples
        if len(g)<3:
            normal=False
            continue

        stat,p=shapiro(g)

        normality_results.append(p)

        # If any group non-normal → use nonparametric test
        if p<alpha:
            normal=False


    print("Normality p-values:",normality_results)


    # ---------- Variance homogeneity ----------
    # Only relevant if data is normal
    equal_var=False

    if normal:

        stat,p=levene(*groups)

        print("Levene test p:",p)

        if p>alpha:
            equal_var=True


    # ---------- Select statistical test ----------
    if normal and equal_var:

        print("Using ANOVA + Tukey HSD")

        # One-way ANOVA
        stat,p=f_oneway(*groups)

        # Tukey posthoc
        tukey=pairwise_tukeyhsd(

            endog=data[y],

            groups=data[x],

            alpha=alpha

        )

        tukey_df=pd.DataFrame(

            tukey._results_table.data[1:],

            columns=tukey._results_table.data[0]

        )

        # Extract significant pairs
        sig_pairs=tukey_df[
            tukey_df.reject==True
        ][['group1','group2','p-adj']].values

        stats_table=tukey_df

        test_used="Using ANOVA + Tukey HSD"


    else:

        print("Using Kruskal-Wallis + Dunn")

        # Non-parametric global test
        stat,p=kruskal(*groups)

        # Dunn posthoc
        dunn=posthoc_dunn(

            data,

            val_col=y,

            group_col=x,

            p_adjust='bonferroni'

        )

        # Convert matrix to pair format
        pairs=dunn.stack().reset_index()

        pairs.columns=['g1','g2','p']

        pairs=pairs[
            (pairs.g1!=pairs.g2) &
            (pairs.p<alpha)
        ]

        # Remove duplicates
        pairs['pair']=pairs.apply(

            lambda r:tuple(
                sorted([r.g1,r.g2])
            ),

            axis=1

        )

        pairs=pairs.drop_duplicates('pair')

        sig_pairs=pairs[['g1','g2','p']].values

        stats_table=dunn

        test_used="Kruskal-Wallis + Dunn"


    print("Global test p:",p)


    # ---------- Plotting ----------
    fig,ax=plt.subplots(figsize=(8,8))


    sns.boxplot(

        data=data,

        x=x,

        y=y,

        palette=palette,

        showfliers=False,
        showmeans=True,
        meanline=True,

        linewidth=2,
        meanprops={
            "color":"black",
            "linestyle":"--",   # ← dashed line
            "linewidth":2
        },
    
        medianprops={
            "color":"black",
            "linewidth":2
        },
        ax=ax

    )


    # Optional datapoints
    if show_points:

        sns.stripplot(

            data=data,

            x=x,

            y=y,

            color='black',

            size=4,

            jitter=0.15,

            ax=ax

        )


    # ---------- Statistical annotation placement ----------
    # Use percentile instead of max to avoid extreme outliers
    ymax=data[y].quantile(.95)

    ymin=data[y].quantile(.05)

    yrange=ymax-ymin

    if yrange==0:
        yrange=1


    bar_base=ymax+yrange*0.15

    spacing=yrange*0.15


    order=list(data[x].unique())


    for i,pair in enumerate(sig_pairs):

        g1,g2,pv=pair

        pos1=order.index(g1)
        pos2=order.index(g2)

        yline=bar_base+i*spacing


        ax.plot(

            [pos1,pos1,pos2,pos2],

            [yline,
             yline+spacing*.3,
             yline+spacing*.3,
             yline],

            lw=2,

            color='black'

        )


        # Convert p-values to stars
        if pv<0.0001:
            stars="****"
        elif pv<0.001:
            stars="***"        
        elif pv<0.01:
            stars="**"
        else:
            stars="*"


        ax.text(

            (pos1+pos2)/2,

            yline+spacing*.35,

            stars,

            ha='center',

            fontsize=14,

            color='red'

        )


    # ---------- Axis scaling ----------
    ax.set_ylim(

        0,

        bar_base+(len(sig_pairs)+2)*spacing

    )


    ax.set_ylabel(ylabel,fontsize=16)

    ax.set_xlabel("")

    ax.set_title(

        title+"\n("+test_used+")",

        fontsize=18

    )


    sns.despine()


    # ---------- Save figure ----------
    if save_path:

        plt.savefig(

            save_path,

            dpi=300,

            bbox_inches='tight'

        )


    plt.show()


    return stats_table , test_used
####################################################################################################


def line_plot(
    data,
    x_col,
    y_col,
    hue_col=None,
    conditions=None,
    filename="LinePlot",
    save_path=".",
    ylabel="",
    xlabel="",
    title="",
    ylim=None,
    show_yticks=True,
    show_xticks=True
):

    """
    Generate a clean seaborn line plot from dataframe columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe

    x_col : str
        Column for X axis

    y_col : str
        Column for Y axis

    hue_col : str, optional
        Column used for grouping lines

    conditions : list, optional
        Subset of hue values to include

    filename : str
        Output filename

    save_path : str
        Directory to save figure

    ylabel : str
        Y axis label

    xlabel : str
        X axis label

    title : str
        Plot title

    ylim : tuple, optional
        Y axis limits (min,max)

    show_yticks : bool
        Show Y tick labels

    show_xticks : bool
        Show X tick labels

    Returns
    -------
    None
    """

    # ==================================================
    # Data filtering
    # ==================================================
    if hue_col and conditions:
        df = data[data[hue_col].isin(conditions)]
    else:
        df = data.copy()

    # ==================================================
    # Figure creation
    # ==================================================
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)

    # ==================================================
    # Line plot
    # ==================================================
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        linewidth=2.5,
        ci=None,
        ax=ax
    )

    # ==================================================
    # Axis styling
    # ==================================================
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ==================================================
    # Y axis settings
    # ==================================================
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.yaxis.set_major_formatter(
        FormatStrFormatter('%.2f')
    )

    if show_yticks:
        ax.tick_params(axis='y', labelsize=14)
    else:
        ax.set_yticklabels([])

    # ==================================================
    # X axis settings
    # ==================================================
    if show_xticks:
        ax.tick_params(axis='x', labelsize=12)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    # ==================================================
    # Grid styling
    # ==================================================
    ax.set_axisbelow(True)

    ax.grid(
        True,
        axis='y',
        linestyle='--',
        linewidth=0.6,
        color='lightgray'
    )

    # ==================================================
    # Labels and title
    # ==================================================
    ax.set_xlabel(
        xlabel,
        fontsize=14,
        fontweight='bold'
    )

    ax.set_ylabel(
        ylabel,
        fontsize=16,
        fontweight='bold'
    )

    ax.set_title(
        title,
        fontsize=18,
        fontweight='bold'
    )

    # ==================================================
    # Legend
    # ==================================================
    if hue_col:
        ax.legend(frameon=False)
    else:
        ax.get_legend().remove()

    # ==================================================
    # Save figure
    # ==================================================
    os.makedirs(save_path, exist_ok=True)

    full_path = os.path.join(
        save_path,
        filename + ".png"
    )

    plt.tight_layout()

    plt.savefig(
        full_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved figure → {full_path}")
###############################################################################################################


def pointplot(
    data,
    x_col,
    y_col,
    hue_col=None,
    conditions=None,
    filename="PointPlot",
    save_path=".",
    ylabel="",
    xlabel="",
    title="",
    y_min=0,
    y_max=2,
    show_yticks=True,
    show_xticks=True
):
    """
    Generate a clean seaborn pointplot from dataframe columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe

    x_col : str
        Column for X axis

    y_col : str
        Column for Y axis

    hue_col : str (optional)
        Column used for grouping (hue)

    conditions : list (optional)
        Subset of hue values to plot

    filename : str
        Output filename

    save_path : str
        Directory to save figure

    ylabel : str
        Y axis label

    xlabel : str
        X axis label

    title : str
        Plot title

    y_min : float
        Minimum Y limit

    y_max : float
        Maximum Y limit

    show_yticks : bool
        Show Y tick labels

    show_xticks : bool
        Show X tick labels

    Returns
    -------
    None
    """

    # ================================
    # Data filtering
    # ================================
    if hue_col and conditions:
        df = data[data[hue_col].isin(conditions)]
    else:
        df = data.copy()

    # ================================
    # Figure creation
    # ================================
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)

    # ================================
    # Point plot
    # ================================
    sns.pointplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        ci=None,
        dodge=True,
        join=True,
        markers="o",
        scale=0.8,
        errwidth=1.2,
        ax=ax
    )

    # ================================
    # Reference line (useful for enrichment plots)
    # ================================
    if y_col.lower() == "enrichment":
        ax.axhline(
            1,
            color="red",
            linestyle="--",
            linewidth=1.2
        )

    # ================================
    # Axis styling
    # ================================
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ================================
    # Y axis settings
    # ================================
    ax.set_ylim(y_min, y_max)

    y_ticks = np.arange(y_min, y_max+0.001, 0.5)
    ax.set_yticks(y_ticks)

    if show_yticks:
        ax.tick_params(axis='y', labelsize=14)
    else:
        ax.set_yticklabels([])

    # ================================
    # X axis settings
    # ================================
    if show_xticks:
        ax.tick_params(axis='x', labelsize=12)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    # ================================
    # Grid styling
    # ================================
    ax.set_axisbelow(True)

    ax.grid(
        True,
        axis='y',
        linestyle='--',
        linewidth=0.6,
        color='lightgray'
    )

    # ================================
    # Labels and title
    # ================================
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')

    ax.set_title(
        title,
        fontsize=18,
        fontweight='bold'
    )

    # ================================
    # Legend
    # ================================
    if hue_col:
        ax.legend(frameon=False)
    else:
        ax.get_legend().remove()

    # ================================
    # Save figure
    # ================================
    os.makedirs(save_path, exist_ok=True)

    full_path = os.path.join(
        save_path,
        filename + ".png"
    )

    plt.tight_layout()

    plt.savefig(
        full_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved figure → {full_path}")