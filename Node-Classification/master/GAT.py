import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn


class GAT_Classification_Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.g1 = tgnn.GATConv(dataset.num_features, 512)
        self.g2 = tgnn.GATConv(512, 256)
        self.g3 = tgnn.GATConv(256, dataset.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.g1(x, edge_index))
        x = self.relu(self.g2(x, edge_index))
        x = self.g3(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Prediction_Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = tgnn.GATConv(in_channels, hidden_channels)
        self.conv2 = tgnn.GATConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # chaining two convolutions with a standard relu activation
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        # cosine similarity
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
