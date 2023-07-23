This folder concludes the code of origin AGCRN, origin STGCN, AGCRN with GFS and STGCN with GFS. [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://arxiv.org/pdf/2007.02842.pdf) [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875)

PEMSD4 and PEMSD8 are available at  [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data). You should first download the data and place them into dir `AGCRN-STGCN/data/`.

## requirements
Python 3.6, Pytorch 1.9, Numpy 1.17, argparse and configparser

## how to play
To replicate the results on PEMSD4 and PEMSD8 datasets, just run the following commands in the "model" folder. 

## STGCN
To replicate the results of STGCN with different adjacency matrices, run
```
python Run.py --model STGCN --adj_type I --use_ln True --device cuda:0
```
You can change `adj_type` according to commands in `stgcn.sh`.
To see the performance of STGCN with GFS, you can run
```
python Run.py --model STGCN --graph_conv_type origin --device cuda:0
```
Also, you may test variants of GFS according the commands in `stgcn.sh`

## AGCRN
The code of AGCRN with different adjacency matrices is dirty, see line 5 to line 38 in `AGCN.py`. You can replace code of `class AVWGCN` with commented lines from line 12 to line 38 and select an adj_mx from line 5 to line 10. Then run
```
python Run.py --model AGCRN --origin True --device cuda:0
```

To test the performance of AGCRN with GFS, you can run
```
python Run.py --model AGCRN --device cuda:0
```
