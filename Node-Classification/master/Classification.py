import torch
from GCN import GCN_Classification_Net, new_GCN_Classification_Net
# from GIN import GIN_Classification_Net
# from GAT import GAT_Classification_Net
from Classification_utils import getMasks, get_weight_decays, load_data, print_model, one_shot_prune, run_pre_train, \
    run_fine_tune, adj_to_dense
import warnings
import time

warnings.filterwarnings('ignore')


def run(dataset_name_, model_name_):
    count = 1000
    x, edge_index, dataset, data, train_mask, test_mask = load_data(dataset_name_)
    start = time.time()
    if model_name == 'GCN':
        edge_index = adj_to_dense(edge_index, len(x))
    end = time.time()
    x = x.cuda()
    # edge_index = torch.ones(len(x),len(x))/len(x)
    edge_index = edge_index.cuda()

    # new_edge_index = new_edge_index.cuda()
    data = data.cuda()
    # if model_name_ == 'GAT':
    #     model = GAT_Classification_Net(dataset)
    # elif model_name_ == 'GIN':
    #     model = GIN_Classification_Net(dataset)
    if model_name_ == 'GCN':
        model = GCN_Classification_Net(dataset)
    # elif model_name_ == 'new_GAT':
    #     model = GIN_Classification_Net(dataset)
    # elif model_name_ == 'new_GIN':
    #     model = GCN_Classification_Net(dataset)
    elif model_name_ == 'new_GCN':
        model = new_GCN_Classification_Net(dataset, len(x))
    model = model.cuda()

    # masks, unmasks = getMasks(model, w_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    ori, best_model, train_time, test_time, test_accs = run_pre_train(dataset, model, optimizer, count, x, edge_index, data,
                                                           train_mask, test_mask, epochs=200,
                                                           adj_length=int(len(x)))
    # # print_model(models)
    # one_shot_prune(best_model, unmasks)
    # new = run_fine_tune(best_model, optimizer, count, x, new_adj, data, weight_decays, masks, unmasks, epochs=200)
    # print_model(models)
    return ori, end - start, train_time, test_time, test_accs  # , new


if __name__ == '__main__':
    dataset_names = ['Coauthor'] # 'PubMed']# , 
    model_names = ['new_GCN']#, 'GCN']

    for dataset_name in dataset_names:
        for model_name in model_names:
            best_acc = 0
            best_test_accs = []
            for _ in range(1):
                start = time.time()
                ori, change_time, train_time, test_time, test_accs = run(dataset_name, model_name)
                end = time.time()
                print(dataset_name, model_name, ori, end - start, change_time, train_time, test_time)
                if best_acc < ori:
                    best_acc = ori
                    best_test_accs = test_accs
            print('best_acc of 10 times training', best_acc)
            # for i in range(len(best_test_accs)):
            #     print(i, best_test_accs[i])
