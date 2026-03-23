<p align="center">
  <img src="images/GA_Mito_Morphology.png" alt="Graphical Abstract">
</p>

<h1 align="center">Mitochondrial Morphotype Characterization</h1>

<p align="center">
  <a href="https://doi.org/10.64898/2026.03.19.712811"><img src="https://img.shields.io/badge/DOI-10.64898%2F2026.03.19.712811-blue" alt="DOI"></a>
  <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue" alt="Python Version">
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

This is the official codebase for the analysis presented in:  
**[Morphotype-Resolved 3D Morphometry Reveals a Structure-Density-Location Coupling in Mitochondrial Networks](https://doi.org/10.64898/2026.03.19.712811)** *(Singh et al., 2026)*.

---

## 🔬 Methods Summary
Our pipeline introduces a morphotype-resolved 3D morphometry approach to quantify mitochondrial networks. By segmenting high-resolution tomograms, we extract distinct spatial and structural features—linking mitochondrial shape (structure), internal characteristics (density), and cellular positioning (location). This multidimensional characterization provides new insights into mitochondrial dynamics and cellular metabolism.

## 🖼️ Example Outputs
This repository generates 3D surface renderings, morphological distributions, and morphotype clustering plots directly from raw tomogram data.

<p align="center">
  <img src="images/G5_1104_9(SML).png" width="400" alt="3D Render">
  <img src="images/Plot.png" width="400" alt="Data Plots">
</p>

---

## ⚙️ Prerequisites & Environment Setup
We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/) or Miniconda to manage dependencies.

**1. Create and activate the environment:**
```bash
conda env create -f env/environment.yml
conda activate sxt_seg
#### Data Requirements & Naming Convention:
- Tomogram file name should end with `_pre_rec.mrc`.
- Mask file name should end with `_pre_rec_labels.mrc`.
- Tomogram and corresponding label must have **identical shape**, e.g. both tomogram and corresponding label has shape `(425, 430, 410)`. Each independent tomogram can be of different size.
- Inside `Data` folder, user have to make individal folders for each tomogram like `Data/Cell1/` `Data/Cell2/`
- To prepare the data for Analysis, copy individual Raw mrc Cell, Mask and corresponding json file inside each `Data/` subfolders.
- Label encoding must follow:
  - `0` → background
  - `1` → cytoplasm label
  - `2` → nucleus
  - `5` → mitochondria
## Analysis
To run the paper experiments and generate plots user should clone this repo, install the given environment and then run the Jupyter notebook `MitoMorph_Analysis.ipynb`
