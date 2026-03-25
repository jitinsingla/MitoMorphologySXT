'''
This file sets paths to data folders:
1. where the Collection is downloaded
2. and where to do all processing
'''
import os
from pathlib import Path

DATA_DIR = r"./Documents"
DATA_DIR = Path(DATA_DIR)

# set working directory for merging files
COLLECTION = Path(DATA_DIR / "Data/") 



