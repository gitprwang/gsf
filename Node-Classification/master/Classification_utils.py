import torch
import numpy as np
import os.path as osp
from torch import Tensor
from L2 import Regularization
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import torch_geometric.transforms as T
from torch_geometric.datasets import BAShapes
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
import time


def sp_A(edge_index, ratio=0.5):
    # ratio代表剩余邻接矩阵的比例
    rm_list = np.random.choice(edge_index.shape[1], int((1 - ratio) * edge_index.shape[1]), replace=False)
    temp = np.array(edge_index)
    temp = np.delete(temp, rm_list, axis=1)
    return torch.from_numpy(temp)


def new_load_data(g_ratio):
    dataset = BAShapes()
    transform = T.Compose([
        T.RandomNodeSplit(num_val=500, num_test=500),
        T.TargetIndegree(),
    ])
    data = dataset[0]
    x, edge_index = data.x, data.edge_index
    new_edge_index = sp_A(edge_index, g_ratio)
    return x, edge_index, new_edge_index, dataset, data


def getMasks(model, w_ratio):
    # w_ratio代表剩余网络参数的比例
    unmasks = []
    masks = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'norms' in name:
                continue
            mask = torch.zeros_like(param)
            unmask = torch.ones_like(param)
            shape0 = mask.shape[0]
            shape1 = mask.shape[1]
            mask = mask.reshape(-1)
            indices = np.random.choice(np.arange(torch.tensor(mask.shape).item()), replace=False,
                                       size=int(torch.tensor(mask.shape).item() * (1 - w_ratio)))
            mask[indices] = 1
            mask = mask.reshape(shape0, shape1)
            unmask = unmask - mask
            masks.append(mask)
            unmasks.append(unmask)
    return masks, unmasks


def get_weight_decays(count):
    base_weight_decay = 5e-4
    add_mul = 1e-7
    weight_decays = [base_weight_decay + add_mul * i for i in range(count)]
    return weight_decays


def load_data(name):
    if name == 'Coauthor':
        dataset = name
        transform = T.Compose([
            T.RandomNodeSplit(num_val=500, num_test=500),
            T.TargetIndegree(),
        ])
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        dataset = Coauthor(root=path, name="Physics")
        data = dataset[0]

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
        dataset.train_mask = a
        dataset.test_mask = b

        x, edge_index = data.x, data.edge_index
        # new_edge_index = sp_A(edge_index, g_ratio)
        return x, edge_index, dataset, data, a, b


    dataset = name
    transform = T.Compose([
        T.RandomNodeSplit(num_val=0, num_test=0.1),
        T.TargetIndegree(),
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=transform)
    data = dataset[0]
    x, edge_index = data.x, data.edge_index
    a = data.train_mask
    b = data.test_mask
    # new_edge_index = sp_A(edge_index, g_ratio)
    return x, edge_index, dataset, data, a, b


def adj_to_dense(adj, length):
    tmp = Tensor.cpu(adj)
    new_tmp = np.array(tmp)
    row = new_tmp[0]
    col = new_tmp[1]
    deletes = []
    for i in range(len(row)):
        if row[i] >= length or col[i] >= length:
            deletes.append(i)
    new_tmp = np.delete(new_tmp, deletes, axis=1)
    temp_data = torch.ones(new_tmp.shape[1])
    # row = torch.from_numpy(row)
    # col = torch.from_numpy(col)
    # tmp = torch.stack((row, col), dim=0)
    new_adj = torch.Tensor(coo_matrix((temp_data, new_tmp), shape=(length, length)).todense())
    return new_adj


def adj_to_sparse(adj, edge_index, length, ratio):
    adj = adj.detach().numpy()
    adj_ = adj.flatten()
    temp_data = np.percentile(adj_, 95, axis=0)  # 后30％
    adj = np.where(adj > temp_data, 1, 0)
    row = []
    col = []
    temp = []
    for i in range(edge_index.shape[1]):
        x = edge_index[0][i]
        y = edge_index[1][i]
        temp.append((int(x.cpu()), int(y.cpu())))
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] == 1:
                if (i, j) in temp:
                    row.append(i)
                    col.append(j)
    choice = np.random.choice(edge_index.shape[1], int(edge_index.shape[1] * ratio), replace=False)
    for i in range(len(choice)):
        x, y = temp[i]
        if x < length and y < length:
            row.append(x)
            col.append(y)
    row = torch.Tensor(np.array(row))
    col = torch.Tensor(np.array(col))
    return torch.stack([row, col], dim=0).long().cuda()


def train(dataset, model, optimizer, count, x, edge_index, data, train_mask, test_mask, adj_length, epochs):
    temp_x = x
    temp_edge_index = edge_index
    ori_length = edge_index.shape[1]
    best_test_acc = 0
    best_model = 0
    train_time = 0
    test_time = 0
    test_accs = []
    for i in range(epochs):
        train_start = time.time()
        model.train()
        optimizer.zero_grad()
        pred = model(temp_x, edge_index)


        loss = F.nll_loss(pred[train_mask], data.y[train_mask])



        loss.backward()
        optimizer.step()
        train_end = time.time()
        test_start = time.time()
        model.eval()
        log_probs = model(x, edge_index)
        test_end = time.time()
        train_acc, test_acc = test(log_probs, temp_x, edge_index, data, train_mask, test_mask)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_model = model
        test_accs.append((train_acc,test_acc))
        # print(best_test_acc)
        train_time = train_time + train_end - train_start
        test_time = test_time + test_end - test_start
    return best_test_acc, best_model, train_time, test_time, test_accs


@torch.no_grad()
def test(log_probs, x, edge_index, data, train_mask, test_mask):
    accs = []
    # for _, mask in data('train_mask', 'test_mask'):
    pred = log_probs[train_mask].max(1)[1]
    acc = pred.eq(data.y[train_mask]).sum().item() / train_mask.sum().item()
    accs.append(acc)

    pred = log_probs[test_mask].max(1)[1]
    acc = pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
    accs.append(acc)
    return accs


def fine_tune(model, optimizer, count, x, new_edge_index, data, weight_decays, masks, unmasks):
    model.train()
    optimizer.zero_grad()
    pred = model(x, new_edge_index)
    loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
    reg_loss = Regularization(model, weight_decays[int(count / 10)], masks, p=2)
    my_reg = reg_loss(model)
    loss = loss + my_reg
    loss.backward()
    optimizer.step()
    count = 0
    for name, param in model.named_parameters():
        with torch.no_grad():
            if 'weight' in name:
                if 'norms' in name:
                    continue
                param[:] = param * unmasks[count]
                count += 1
    return loss.item()


def print_model(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'norms' in name:
                continue
            print(param)


def one_shot_prune(model, unmasks):
    my_count = 0
    for name, param in model.named_parameters():
        with torch.no_grad():
            if 'weight' in name:
                if 'norms' in name:
                    continue
                param[:] = param * unmasks[my_count]
                my_count += 1


def run_pre_train(dataset, model, optimizer, count, x, edge_index, data, train_mask, test_mask, epochs,
                  adj_length):
    best_test_acc, best_model, train_time, test_time, test_accs = train(dataset, model, optimizer, count, x, edge_index, data,
                                                             train_mask,
                                                             test_mask,
                                                             adj_length, epochs)
    return best_test_acc, best_model, train_time, test_time, test_accs


def run_fine_tune(model, optimizer, count, x, new_edge_index, data, weight_decays, masks, unmasks, epochs):
    # print("fine_tune...")
    best_test_acc = 0
    for epoch in range(epochs):
        loss = fine_tune(model, optimizer, count, x, new_edge_index, data, weight_decays, masks, unmasks)
        train_acc, test_acc = test(model, x, new_edge_index, data)
        if best_test_acc < test_acc:
            best_test_acc = test_acc
        # if epoch % 10 == 0:
        #     print('In epoch {:3}  loss: {:.3f}  test acc: {:.3f}  best acc {:.3f}'.format(
        #         epoch, float(loss), float(test_acc), float(best_test_acc)))
    return best_test_acc
