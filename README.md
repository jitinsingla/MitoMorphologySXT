# Mitochondrial Morphotype Characterization
This is the codebase used for the analysis in paper. [Yadav, A., Singh, A., Deshmukh, A., Varma, R., Singh, A., White, K., & Singla, J. (2026). Morphotype-Resolved 3D Morphometry Reveals a Structure-Density-Location Coupling in Mitochondrial Networks](https://doi.org/10.64898/2026.03.19.712811)

## Prerequisties / Environment Setup
Install Anaconda for setup ([link](https://docs.anaconda.com/anaconda/install/))

#### Create environment (recommended):
```
conda env create -f env/environment.yml
conda activate sxt_seg
```
#### Data preparation:
Raw mrc file name should end with `_pre_rec.mrc`.
Mask mrc file name should end with `_pre_rec_labels.mrc`.
To prepare the data for Analysis, copy raw mrc files, Masks and corresponding json files in `Data/File(s)/`.
