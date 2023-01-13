from torch_geometric.datasets import Reddit
import os.path as osp
import numpy as np
import torch

dataset = 'Reddit'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Reddit(root=path)

length = len(dataset[0].x)
train_length = int(length * 0.7)
train_mask = np.random.choice(length, train_length, replace=False)
a = torch.zeros(length, dtype=torch.bool)
b = torch.zeros(length, dtype=torch.bool)
for i in range(len(a)):
    if i in train_mask:
        a[i] = True
for i in range(len(b)):
    if i not in train_mask:
        b[i] = True
dataset.train_mask=a
dataset.test_mask=b
print(dataset[0])