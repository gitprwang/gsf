
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from AGCRN import AGCRN as Network
from BasicTrainer import Trainer
# from Trainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters


#*************************************************************************#
Mode = 'train'
DEBUG = 'False'
DATASET = 'PEMSD4'      #PEMSD4, metr-la, pems-bay, PEMSD7, PEMSD3 or PEMSD8 
DEVICE = 'cuda:0'
MODEL = 'AGCRN'

#get configuration
config_file = './{}_{}.conf'.format(DATASET, MODEL)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args.add_argument('--tr', default=1.0, type=float)
args.add_argument('--enable_cuda', type=bool, default='True', help='enable CUDA, default as True')
# args.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
# args.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
args.add_argument('--n_his', type=int, default=config['data']['lag'])
args.add_argument('--n_pred', type=int, default=config['data']['horizon'], help='the number of time interval for predcition, default as 3')
args.add_argument('--time_intvl', type=int, default=5)
args.add_argument('--Kt', type=int, default=3)
args.add_argument('--stblock_num', type=int, default=2)
args.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
args.add_argument('--Ks', type=int, default=3, choices=[3, 2])
args.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv')
args.add_argument('--gso_type', type=str, default='sym_renorm_adj', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
args.add_argument('--enable_bias', type=bool, default=True, help='default as True')
args.add_argument('--droprate', type=float, default=0.)
args.add_argument('--lr', type=float, default=0.001, help='learning rate')
args.add_argument('--weight_decay_rate', type=float, default=0, help='weight decay (L2 penalty)')
# args.add_argument('--batch_size', type=int, default=32)
# args.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
args.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
args.add_argument('--step_size', type=int, default=10)
args.add_argument('--gamma', type=float, default=0.95)
args.add_argument('--patience', type=int, default=30, help='early stopping patience')
args.add_argument('--origin', type=bool, default=False)
args.add_argument('--adaptive', type=bool, default=False)
args.add_argument('--use_trick', type=bool, default=False)
args.add_argument('--use_ln', type=bool, default=False)
args.add_argument('--finetune', type=bool, default=False)
args.add_argument('--finetune_scale', type=float, default=10.0)
args.add_argument('--shuffle', type=bool, default=True)
args.add_argument('--adj_type', type=str, default='None')
args = args.parse_args()
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

# init model
# agcrn
from get_stgcn_model import get_model
from stgcn_origin import get_origin_stgcn, cal_adj

if args.model=='AGCRN':
    model = Network(args)
if args.model=='STGCN':
    if args.origin:
        import torch.nn.functional as F
        print('------------------------------------------')
        print('model: STGCN origin\n adj_type '+args.adj_type)
        print('------------------------------------------')
        node_num = args.num_nodes
        if args.adj_type=='I':
            adj_mx = torch.eye(node_num)
        elif args.adj_type=='mean':
            adj_mx = torch.ones(node_num, node_num)/node_num
        elif args.adj_type=='I+mean':
            adj_mx = torch.eye(node_num)+1.0/node_num
        elif args.adj_type=='rand':
            adj_mx = F.softmax(torch.randn(node_num, node_num))
        elif args.adj_type=='I+rand':
            adj_mx = torch.eye(node_num)+F.softmax(torch.randn(node_num, node_num))
        elif args.adj_type=='I-mean':
            adj_mx = torch.eye(node_num)-1.0/node_num
        else:
            adj_mx = cal_adj('../data/PeMSD8/distance.csv', args.num_nodes)
        adj_mx = adj_mx.to(args.device)
        model = get_origin_stgcn(args, adj_mx)
    else:
        model = get_model(args)


model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1 and not (torch.numel(p)==torch.sum(p)):
        nn.init.xavier_uniform_(p)
    # else:
    #     nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)
def loss_func(pred, target):
    #init loss function, optimizer
    if args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError
    return loss(pred, target)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
if args.model=='STGCN':
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay_rate, amsgrad=False)

#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, args.model+args.adj_type)
args.log_dir = log_dir

#start training
trainer = Trainer(model, loss_func, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)






if args.mode == 'train':
    trainer.train()
    if args.model == 'AGCRN':
        np.save('{}_embed.npy'.format(args.embed_dim), model.node_embeddings.detach().cpu().numpy())
elif args.mode == 'test':
    # def loader(n, batches=100, batch_size=32):
    #     test_x_tensor = torch.randn([batches*batch_size, 12, n, 1]).type(torch.FloatTensor).to(args.device)
    #     test_target_tensor = torch.randn([batches*batch_size, 12, n, 1]).type(torch.FloatTensor).to(args.device)
    #     test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    #     return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = loader(args.num_nodes)
    # from time import time
    # # model.load_state_dict(torch.load('../pre-trained/{}.pth'.format(args.dataset)))
    # print("testing model")
    # print(len(test_loader))
    model = torch.load("./experiments/{}/best_model.pth".format(args.dataset)).to(args.device)
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
    # model.eval()
    # costs = []
    # for i in range(10):
    #     with torch.no_grad():
    #         b = time()
    #         for batch_idx, (data, target) in enumerate(test_loader):
    #             output = model(data)
    #         time_cost = time()-b
    #         costs.append(time_cost)
    # costs = np.array(costs)
    # print(costs)
    # print(costs.mean())
else:
    raise ValueError
