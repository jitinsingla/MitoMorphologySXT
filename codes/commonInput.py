import importlib, json, pprint, os, sys, gc, mrcfile, glob
import numpy as np
from utils import *
import pandas as pd
import seaborn as sns
sys.path.append("../")
import path_settings
from pprint import pprint
from copy import deepcopy
from image_properties import *
import matplotlib.pyplot as plt


jsonFileList, LabelMRCfilelist, RawMRCfilelist = sortedFileList()

with open("/home/anshu/Documents/Github_push/Data/parameters.json") as f:
    op = json.load(f)
   
cellNameDict = {}
for cond in op["conditions"]:
    if cond not in cellNameDict:
        cellNameDict[cond] = [_+"_pre_rec" for _ in op[cond]]
        
cell_names = [ImageData(_).data['name'] for _ in jsonFileList]
conditions = {}
for cond in op["conditions"]: 
    for cell_ID in op[cond]: 
        conditions[cell_ID] = cond 
cell_IDs = list(conditions.keys())
for cell_name in cell_names:  
    conditions[cell_name] = {'condition': 'Not-INS-1E'} 
    for cell_ID in cell_IDs:  
        # print(cell_ID)
        if cell_name.find(cell_ID)>0: 
            conditions[cell_name] = {'condition':conditions[cell_ID], 'cell':cell_ID} 
            del conditions[cell_ID]

condition_order = {
    'No_Stimulation': 0,
    'Glu(5min)': 1
}
sorted_data = sorted(conditions.items(), key=lambda x: (condition_order[x[1]['condition']], x[1]['cell']))
conditions = dict(sorted_data)
