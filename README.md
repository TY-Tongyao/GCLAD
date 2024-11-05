# Graph-Enhanced Multi-Scale Contrastive Learning for Graph Anomaly Detection with Adaptive Diffusion Models


This is the PyTorch source code for the GCLAD. 
The code runs on Python 3. 
Install the dependencies and prepare the datasets with the following commands:



## Dataset

The five public datasets used in the paper are shown below.

The datasets can be downloaded from the following:

### Cora Dataset
The Cora dataset comes from the link below: https://paperswithcode.com/dataset/cora

### Citeseer Dataset
The Citeseer dataset comes from the link below: https://paperswithcode.com/dataset/citeseer

### Pubmed Dataset
The Pubmed dataset comes from the link below: https://paperswithcode.com/dataset/pubmed

### Flickr Dataset
The Flickr dataset comes from the link below: https://brightdata.com/products/datasets/image/flickr

### Blogcatalog Dataset
The Blogcatalog dataset comes from the link below: https://www.kaggle.com/datasets/pfluoo/blogcatalog



### Requirements

The proposed GCLAD is implemented with python 3.7 on a NVIDIA 3070 GPU. 

Python package information is summarized in **requirements.txt**:

- torch==1.10.2
- dgl==0.4.1
- numpy==1.19.2

### Quick Start

python run.py
