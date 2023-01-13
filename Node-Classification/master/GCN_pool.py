import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from Classification_utils import load_data


class GCN_POOL(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.g1 = tgnn.GCNConv(dataset.num_features, 512)
        self.g2 = tgnn.GCNConv(512, 256)
        self.g3 = tgnn.GCNConv(256, dataset.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, length, dim):
        g1 = tgnn.GCNConv(length, int(dim/4))
        g2 = tgnn.GCNConv(int(dim/4), int(dim/8))
        g3 = tgnn.GCNConv(int(dim/8), dim)

        g4 = tgnn.GCNConv(length, dim)
        # x = self.relu(g1(x, edge_index))
        # x = self.relu(g2(x, edge_index))
        # x = g3(x, edge_index)
        x = self.relu(g4(x, edge_index))
        return F.softmax(x, dim=1)


if __name__ == '__main__':
    x, edge_index, new_edge_index, dataset, data = load_data('Cora', 0.2)
    model = GCN_POOL(dataset)
    for i in range(10):
        dim = i * 10
        pred = model(x, edge_index, dataset, dim)
        print(pred.shape)
