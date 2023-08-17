# SEA-GWNN: Simple and Effective Generalized Adaptive Graph Wavelet Neural Network
This is a pytorch implementation of the paper titled: "SEA-GWNN: Simple and Effective Adaptive Graph Wavelet Neural Network". 

## Environment Settings    
- pytorch 1.12.0
- numpy 1.23.4
- torch-geometric 1.7.2
- tqdm 4.64.1
- scipy 1.9.3
- seaborn 0.12.0
- scikit-learn 1.1.3
- ogb 1.3.5

All the requirements are given inside the "requirements.txt" file.
## Datasets
We provide the Cora and Chameleon datasets in the folder 'data/' and 'new_data/' direactory. Additionally, you can also download all the datasets utilized in this paper form the Pei et. al. 2018, "Geom-GCN: Geometric Graph Convolutional Networks".

## Code Structure
The folder "homophilic graphs" is the code for the for standard citation networks (Cora, Citeseer, PubMed); and the folder "heterophilic graphs" is the code for the results in all the heterophilic datasets. Finally "large graphs" contain code for the large scale graphs.


## Environment Setup
You will require pytroch gpu version 1.12.0 from pytorch (https://pytorch.org/get-started/previous-versions/).
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

```
Then run requirements.txt file.
```
pip install -r requirements.txt
```
