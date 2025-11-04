# Repository Description  
This repository contains the model codes for the paper:  
​**​《d-band center-guided high-fidelity generative model for inverse materials design》**  
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cej.2025.169918-blue?labelColor=555555&style=flat&logoColor=white)](https://doi.org/10.1016/j.cej.2025.169918)  
If the paper is useful to you, kindly consider citing it.

# Model Schematic Diagram
![Model](https://github.com/jiahao-codes/dBandDiff/blob/0c694336cb502f4c42770611cdc0dafda8ef6bae/pic/Model%20Diagram.png)

# Overview
![Model](https://github.com/jiahao-codes/dBandDiff/blob/c4c5e28496c7575ac9760584b3f8ae1fe5a7384d/pic/2f9a9812ae9aa023943b76213fbe67e0.jpeg)

# Model Training and Inference

## Create a Conda Environment and Install Dependencies

    conda create -n dbanddiff python=3.8 -y
    conda activate dbanddiff
    pip install -r requirements.txt
    
## Training

### Step 1: Extract the Dataset

    unzip dataset.zip

### Step 2: Run the **​[Training.ipynb](https://github.com/jiahao-codes/dBandDiff/blob/deebf70aec57daf4b683dcbbcdcc05f8228fe8f1/Training.ipynb)​**

    conda activate dbanddiff
    jupyter notebook Training.ipynb
    
## Inference

### Step 1: Download the model weights from Google Drive and place it in the same directory as "Generation.ipynb": 
https://drive.google.com/file/d/1fs1_qkx5HE40SU5xnBBHoVERTCKYt454/view?usp=sharing
    
### Step 2: Run the **​[Generation.ipynb](https://github.com/jiahao-codes/dBandDiff/blob/deebf70aec57daf4b683dcbbcdcc05f8228fe8f1/Generation.ipynb)​**​
    conda activate dbanddiff
    jupyter notebook Generation.ipynb


# Dependency  
Including **​[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)​**​, **​[PyTorch](https://github.com/pytorch/pytorch)​**​, **​[Pymatgen](https://github.com/materialsproject/pymatgen)​**, etc. Please refer to the **​[requirement.txt](https://github.com/jiahao-codes/dBandDiff/blob/cae010a74b32716a3d1cd047faf3c6ba6cf39d3d/requirements.txt)​**​ for details.

# Support  
For any questions, please raise issues or contact wogaho1999@gmail.com.
