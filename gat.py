from datetime import datetime
from Utils import save_x, com_top_K_patter_x, sup_pattern, save_wnode, get_grapg_x, save_gat
from model.gin import GIN
from model.nodeweight import get_node_weight
from model.sage import GraphSAGE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import GAT


# from until import save_x
def load_x_data(adj):
    feature_counts = {}
    feature_lable = {}
    file_name = 'Data/twitter.txt'
    count = 0
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('e'):
                parts = line.split()
                if len(parts) >= 3:
                    row = int(parts[1]) - 1
                    col = int(parts[2]) - 1
                    adj[row, col] = 1
                    adj[col, row] = 1
            elif line.startswith('v'):
                parts = line.split()
                if len(parts) >= 3:
                    feature = int(parts[2])
                    feature_lable[count] = int(parts[2])
                    count += 1
                    if feature not in feature_counts:
                        feature_counts[feature] = 1
                    else:
                        feature_counts[feature] += 1

    nclass = len(feature_counts)
    print(nclass)
    return nclass, adj, feature_lable


def load_x_data_graph(file_name, edge_num):
    feature_lable = {}
    count_edge = 0
    count = 0
    file_name = 'Data/' + file_name + '.txt'
    edge_index = np.empty((2, edge_num * 2))
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('e'):
                parts = line.split()
                if len(parts) >= 3:
                    row = int(parts[1])
                    col = int(parts[2])
                    edge_index[0][count_edge] = row
                    edge_index[1][count_edge] = col
                    count_edge += 1
                    edge_index[0][count_edge] = col
                    edge_index[1][count_edge] = row
                    count_edge += 1
            elif line.startswith('v'):
                parts = line.split()
                if len(parts) >= 3:
                    feature_lable[count] = int(parts[2])
                    count += 1

    edge_index_torch = torch.from_numpy(edge_index).to(torch.long)
    return feature_lable, edge_index_torch


def graph_to_x(i):
    # amazon_table>2
    feature_lable_dict = {}
    start = datetime.now()
    nhid = 32
    dropout = 0.5
    file_name = file_name_dict[i]
    v_num = v_num_dict[i]
    edge_num = edge_num_dict[i]
    nfeatuer = featuer_num_dict[i]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feature_lable, edge_index = load_x_data_graph(file_name, edge_num)
    x = torch.zeros(v_num, nfeatuer, )
    for j in range(v_num):
        x[i][feature_lable[i] - 1] = 1
    data = Data(x=x, edge_index=edge_index)
    sage_time = datetime.now()
    # print(data)
    if gnn_model == 'SAGE':
        model = GraphSAGE(num_features=data.num_features, num_classes=nfeatuer)
    if (gnn_model == 'GAT'):
        model = GAT(num_features=data.num_features, num_classes=nfeatuer)
    if (gnn_model == 'GIN'):
        model = GIN(num_features=data.num_features, num_classes=nfeatuer, dropout=dropout)
    # model = GraphSAGE(num_features=data.num_features, num_classes=nfeatuer)
    # model = GAT(num_features=data.num_features, num_classes=nfeatuer)
    # model = GIN(num_features=data.num_features, num_classes=nfeatuer, dropout=dropout)
    output = model(data.x, data.edge_index)

    # save_x(output, file_name)
    end = datetime.now()
    print("---------------------")
    return output


def get_graph_x(file_name):
    x = np.zeros((v_num, nfeatuer), dtype=np.float64)
    with open("data/grapg_x/" + file_name, 'r') as file:
        for line in file:
            numlist = line.split()
            for i in range(v_num):
                for j in range(nfeatuer):
                    x[i][j] = numlist[j]

    return x


# file_name_dict =["twitter","mico","patent","dblp","twitch"]
# v_num_dict=[81306,100000,2745761,425957,168114]
# edge_num_dict=[2420766,1080298,13965409,1049870,6797557]
# featuer_num_dict=[25,29,37,60,79]

now = datetime.now()

file_name_dict = ["mico"]
v_num_dict = [100000]
edge_num_dict = [1080298]
featuer_num_dict = [29]
file_name = ""
v_num = 0
edge_num = 0
nfeatuer = 0
# ks=[300,400,500]
ks = [200]

if __name__ == '__main__':
    gnn_models = ["GIN"]
    for gnn_model in gnn_models:
        for i, file_name in enumerate(file_name_dict):
            feature_lable_dict = {}
            start = datetime.now()
            # dropout = 0.5
            file_name = file_name_dict[i]
            v_num = v_num_dict[i]
            edge_num = edge_num_dict[i]
            nfeatuer = featuer_num_dict[i]
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            feature_lable, edge_index = load_x_data_graph(file_name, edge_num)
            x = torch.zeros(v_num, nfeatuer)
            # x = torch.randn(v_num, nfeatuer)
            for ii in range(v_num):
                    # print(feature_lable[ii])
                    x[ii][feature_lable[ii] - 1] = 1

            for ii in range(20):
                print(x[ii])
            # save_gat(x, file_name, gnn_model)



