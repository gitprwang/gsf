import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from model.STGCN import *

#import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for multiple GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parameters(args):
    # print('Training configs: {}'.format(args))

    # For stable experiment results
    SEED = args.seed
    set_env(SEED)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    
    n_his = args.n_his

    Kt = args.Kt
    stblock_num = args.stblock_num
    Ko = n_his - (Kt - 1) * 2 * stblock_num
    if args.act_func == 'glu' or args.act_func == 'gtu':
        act_func = args.act_func
    else:
        raise NotImplementedError(f'ERROR: {args.act_func} is not defined.')
    Ks = args.Ks
    graph_conv_type = args.graph_conv_type

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    enable_bias = args.enable_bias
    droprate = args.droprate
    
    return device, n_his, Kt, act_func, Ks, graph_conv_type, blocks, enable_bias, droprate



def prepare_model(Kt, Ks, blocks, n_his, n_vertex, act_func, graph_conv_type, gso, enable_bias, droprate, device):
    model = STGCNGraphConv(Kt, Ks, blocks, n_his, n_vertex, act_func, graph_conv_type, gso, enable_bias, droprate).to(device)
    return model

def get_model(args):
    n_vertex = args.num_nodes
    device, n_his, Kt, act_func, Ks, graph_conv_type, blocks, enable_bias, droprate = get_parameters(args)
    return prepare_model(Kt, Ks, blocks, n_his, n_vertex, act_func, graph_conv_type, None, enable_bias, droprate, device) 
