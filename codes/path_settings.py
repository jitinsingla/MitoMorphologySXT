'''
This file sets paths to data folders:
1. where the Collection is downloaded
2. and where to do all processing
'''
import os
from pathlib import Path

#DATA_DIR_1 = r"/home/anshu/Documents/Github_push"
#DATA_DIR_1 = Path(DATA_DIR_1)
DATA_DIR = r"/home/anshu/Documents/Github_push"

DATA_DIR = Path(DATA_DIR)

# set working directory for merging files
COLLECTION = Path(DATA_DIR / "Data/") # "Original_data/"

#RAW = Path(DATA_DIR_1 / "raw_tomogram_mrc_files/")
#RAW_LAC = Path(DATA_DIR_1 / "LACfactorXraw_tomogram_mrc_files/")

