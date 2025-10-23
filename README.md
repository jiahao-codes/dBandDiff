# Repository Description  
This repository contains the model codes for the paper:  
​**​《d-band center-guided high-fidelity generative model for inverse materials design》**  
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cej.2025.169918-blue?labelColor=555555&style=flat&logoColor=white)](https://doi.org/10.1016/j.cej.2025.169918)  
If the code is useful to you, please consider citing it.

# Model Training and Inference
## Training

### Step 1: Extract the Dataset

    unzip dataset.zip

### Step 2: Create a Conda Environment and Install Dependencies

    conda create -n dbanddiff-training python=3.8 -y
    conda activate dbanddiff-training
    pip install -r requirements.txt

### Step 3: Run the **​[Training.ipynb](https://github.com/jiahao-codes/dBandDiff/blob/deebf70aec57daf4b683dcbbcdcc05f8228fe8f1/Training.ipynb)​**​ in Jupyter Notebook

    jupyter notebook
    
## Inference
### Step 1: Create a Conda Environment and Install Dependencies

    conda create -n dbanddiff-inference python=3.8 -y
    conda activate dbanddiff-inference
    pip install -r requirements.txt

### Step 2: Download the model weights from Google Drive and place it in the same directory as "Generation.ipynb": https://drive.google.com/file/d/1fs1_qkx5HE40SU5xnBBHoVERTCKYt454/view?usp=sharing
    
### Step 3: Run the **​[Generation.ipynb](https://github.com/jiahao-codes/dBandDiff/blob/deebf70aec57daf4b683dcbbcdcc05f8228fe8f1/Generation.ipynb)​**​ in Jupyter Notebook

    jupyter notebook
    
# Model Schematic Diagram
![Model](https://github.com/jiahao-codes/dBandDiff/blob/0c694336cb502f4c42770611cdc0dafda8ef6bae/pic/Model%20Diagram.png)

# Dependency  
Including **​[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)​**​, **​[PyTorch](https://github.com/pytorch/pytorch)​**​, **​[Pymatgen](https://github.com/materialsproject/pymatgen)​**, etc. Please refer to the **​[requirement.txt](https://github.com/jiahao-codes/dBandDiff/blob/cae010a74b32716a3d1cd047faf3c6ba6cf39d3d/requirements.txt)​**​ for details.

# Support  
For any questions, please raise issues or contact wogaho1999@gmail.com.
