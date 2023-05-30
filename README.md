# SGAD
This is a PyTorch implementation of our proposed SGAD algorithm.

## Requirements
* python3.8
* pytorch==1.13.1
* torch-scatter==2.1.0
* torch-sparse==0.6.15
* torch-cluster==1.6.0
* torch-geometric==2.4.0

## Datasets
Datasets will be downloaded automatically via Pytorch Geometric when running the codes. You can refer [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) for more details. 


## Quick Start:
Just execuate the following command:
```
python main.py
```
Supported datasets:
* `BZR`, `COX2`, `PROTEINS`, `AIDS`, `NCI1`, `NCI109`. 
