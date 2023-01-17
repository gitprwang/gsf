import torch
import torch.nn.functional as F
import torch.nn as nn

# adj_mx = torch.eye(307)
# adj_mx = torch.eye(307)+1.0/307
# adj_mx = torch.eye(307)-1.0/307
# adj_mx = torch.ones(307, 307)/307
# adj_mx = F.softmax(torch.randn(307, 307))
# adj_mx = torch.eye(307)+F.softmax(torch.randn(307, 307))

# class AVWGCN(nn.Module):
#     def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
#         super(AVWGCN, self).__init__()
#         self.cheb_k = cheb_k
#         self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
#         self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
#     def forward(self, x, node_embeddings):
#         #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
#         #output shape [B, N, C]
#         node_num = node_embeddings.shape[0]

#         # supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
#         # supports = supports.to(node_embeddings.device)

#         supports = adj_mx.to(node_embeddings.device)

#         support_set = [torch.eye(node_num).to(supports.device), supports]
#         #default cheb_k = 3
#         for k in range(2, self.cheb_k):
#             support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
#         supports = torch.stack(support_set, dim=0)
#         weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
#         bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
#         x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
#         x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
#         x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
#         return x_gconv

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, node_num):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, dim_out))
        
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        # supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        supports = (torch.eye(node_num) + 1./node_num).to(x.device)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

def func(x):
    return x

class MLPGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, node_num):
        super(MLPGCN, self).__init__()
        self.in_dim = dim_in
        self.out_dim = dim_out 
        self.act = torch.relu
        self.align = nn.Linear(dim_in, dim_out) if dim_in!=dim_out else func
        self.ln = nn.LayerNorm([node_num, dim_out])
        self.w = nn.Parameter(torch.randn(self.in_dim, self.out_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(self.out_dim), requires_grad=True)

        self.agg = nn.Parameter(torch.ones(node_num, 1), requires_grad=True)

    def forward(self, x, node_embeddings):
        # x.shape : B,N,d
        res = self.align(x)
        x = torch.einsum('acd,de->ace', x, self.w)
        x = x + self.b
        
        x = x*self.agg

        return res + self.ln(self.act(x))

# def func(x):
#     return x


# class MLPGCN(nn.Module):
#     def __init__(self, dim_in, dim_out, node_num):
#         super(MLPGCN, self).__init__()
#         self.in_dim = dim_in
#         self.out_dim = dim_out
#         self.act = torch.relu
#         self.align = nn.Linear(dim_in, dim_out) if dim_in != dim_out else func
#         self.ln = nn.LayerNorm([node_num, dim_out])
#         self.w = nn.Parameter(torch.randn(self.in_dim, self.out_dim), requires_grad=True)
#         self.b = nn.Parameter(torch.zeros(self.out_dim), requires_grad=True)

#         self.agg = nn.Parameter(torch.ones(node_num, 1), requires_grad=True)

#     def forward(self, x, adj=None):
#         # x.shape : B,N,d
#         res = self.align(x)
#         x = torch.einsum('cd,de->ce', x, self.w)
#         x = x + self.b
#         x = x*self.agg
#         return res + self.ln(self.act(x))
