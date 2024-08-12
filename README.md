# Continuous SO(3) Equivariant Convolution for 3D Point Cloud Analysis [ECCV 2024]
<img src="media/fig_3_proto_1.png">

## About
This repository contains the implementation of Continuous SO(3) Equivariant Convolution (CSEConv) and reproducible experiments from the corresponding paper.

## Installation & Dependencies
```bash
# Highly recommend to install PyTorch and PyTorch3D libraries using conda
conda create -n cse python=3.9
conda activate cse

# configure PyTorch installment with your own environment
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
conda install -c iopath iopath
conda install pytorch3d -c pytorch3d

# Install dependencies to use training & evaluation codes
pip install tqdm omegaconf wandb matplotlib scikit-learn
```

## Dataset Configuration
* Download ModelNet40 dataset.
    * https://github.com/antao97/PointCloudDatasets
    * Download `modelnet40_hdf5_2048.zip` from the referenced link.
* Download ScanObjectNN dataset.
    * https://hkust-vgd.github.io/scanobjectnn/
    * Follow the instruction and download `h5_files.zip`.
* Create directory `dataset` and unzip datasets.
    ```bash
    cd ~/CSEConv
    # download zipped files 
    mkdir dataset
    ```

## How to use

## Reproducible Weights

## Citation
If you find this repository is useful for your project, please consider to cite it as belows.
