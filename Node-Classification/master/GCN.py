import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.nn import global_mean_pool


def func(x):
    return x


class MLPGCN(nn.Module):
    def __init__(self, dim_in, dim_out, node_num):
        super(MLPGCN, self).__init__()
        self.in_dim = dim_in
        self.out_dim = dim_out
        self.act = torch.relu
        self.align = nn.Linear(dim_in, dim_out) if dim_in != dim_out else func
        self.ln = nn.LayerNorm([node_num, dim_out])
        self.w = nn.Parameter(torch.randn(self.in_dim, self.out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(self.out_dim), requires_grad=True)

        self.agg = nn.Parameter(torch.ones(node_num, 1), requires_grad=True)

    def forward(self, x, adj=None):
        # x.shape : B,N,d
        res = self.align(x)
        x = torch.einsum('cd,de->ce', x, self.w)
        x = x + self.b
        x = x*self.agg
        return res + self.ln(self.act(x))


class gcn_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(gcn_layer, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))
        self.ln = nn.LayerNorm([out_dim])
        self.align = nn.Linear(in_dim, out_dim) if in_dim != out_dim else func

    def forward(self, x, adj):
        # x:B N D
        # adj: N N
        res = self.align(x)
        x = adj.matmul(x)
        x = x.matmul(self.w) + self.b
        return res + self.ln(torch.tanh(x))


class GCN_Classification_Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.g1 = gcn_layer(dataset.num_features, 512)
        self.g2 = gcn_layer(512, 256)
        self.g3 = gcn_layer(256, dataset.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.g1(x, edge_index))
        x = self.relu(self.g2(x, edge_index))
        x = self.g3(x, edge_index)
        return F.log_softmax(x, dim=1)


class new_GCN_Classification_Net(torch.nn.Module):
    def __init__(self, dataset, node_num):
        super().__init__()
        self.g1 = MLPGCN(dataset.num_features, 512, node_num)
        self.g2 = MLPGCN(512, 256, node_num)
        self.g3 = MLPGCN(256, dataset.num_classes, node_num)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.g1(x, edge_index))
        x = self.relu(self.g2(x, edge_index))
        x = self.g3(x, edge_index)
        return F.log_softmax(x, dim=1)

# class GCN_Prediction_Net(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = tgnn.GCNConv(in_channels, hidden_channels)
#         self.conv2 = tgnn.GCNConv(hidden_channels, out_channels)
#
#     def encode(self, x, edge_index):
#         # chaining two convolutions with a standard relu activation
#         x = self.conv1(x, edge_index).relu()
#         return self.conv2(x, edge_index)
#
#     def decode(self, z, edge_label_index):
#         # cosine similarity
#         return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
#
#     def decode_all(self, z):
#         prob_adj = z @ z.t()
#         return (prob_adj > 0).nonzero(as_tuple=False).t()
#
#
# class GCN_Graph_Classification(torch.nn.Module):
#     def __init__(self, dataset, hidden_channels):
#         super(GCN_Graph_Classification, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = tgnn.GCNConv(dataset.num_node_features, hidden_channels)
#         self.conv2 = tgnn.GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = tgnn.GCNConv(hidden_channels, hidden_channels)
#         self.lin = nn.Linear(hidden_channels, dataset.num_classes)
#
#     def forward(self, x, edge_index, batch):
#         # 1. 获得节点嵌入
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
#
#         # 2. Readout layer
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#
#         # 3. 分类器
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#
#         return x
