from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from tqdm import tqdm

class GraphMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes=307, act=torch.relu):
        super(GraphMLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.act = act
        self.w = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        self.ln = nn.LayerNorm([num_nodes, out_dim])

    def forward(self, x):
        # x.shape : B,T,N,d
        x = torch.einsum('abcd,de->abce', x, self.w)
        x = x + self.b
        x = self.act(x)
        x = self.ln(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes=307, act=torch.relu):
        super(GCN, self).__init__()
        self.w = nn.Parameter(torch.randn(in_dim, out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(out_dim), requires_grad=True)
        self.act = act
        
    def forward(self, x, adj):
        x = torch.einsum('abcd,de->abce', x, self.w)
        x = x + self.b
        x = torch.matmul(adj, x)
        x = self.act(x)
        return x


def test(num_nodes=307, dim=64):
    x = torch.randn(64, 12, num_nodes, dim).cuda()
    adj = torch.randn(num_nodes,num_nodes).cuda()
    mlp = GraphMLP(dim, dim, num_nodes).cuda()
    gcn = GCN(dim, dim, num_nodes).cuda()
    b = time()
    for _ in range(1000): # tqdm(range(1000)):
        mlp(x)
    print('mlp time on nodes {} is {}:'.format(num_nodes, time()-b))

    b = time()
    for _ in range(1000): # tqdm(range(1000)):
        gcn(x, adj)
    print('gcn time on nodes {} is {}:'.format(num_nodes, time()-b))

if __name__=='__main__':
    for node in range(100,2000,100):
        test(node)
