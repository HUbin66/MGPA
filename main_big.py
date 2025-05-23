import time
from datetime import datetime
import tracemalloc
import time

from memory_profiler import profile
from Utils import save_x, com_top_K_patter_x, sup_pattern, save_wnode, get_grapg_x, save_gat
from node_weight import get_wnode
from model.gat import GAT
from model.gcn import GNN
from model.gin import GIN
from model.nodeweight import get_node_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.sage import GraphSAGE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import  GCN
# from until import save_x
from torch_geometric.loader import DataLoader

# from run import run


def load_x_data(adj):
    feature_counts = {}
    feature_lable = {}
    file_name = 'Data/twitter.txt'
    count=0
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
                    feature_lable[count]=int(parts[2])
                    count+=1
                    if feature not in feature_counts:
                        feature_counts[feature] = 1
                    else:
                        feature_counts[feature] += 1

    nclass = len(feature_counts)
    print(nclass)
    return nclass,adj,feature_lable
def load_x_data_graph(file_name,edge_num,y):
    feature_lable = {}
    count_edge = 0
    # count =
    file_name='Data/'+file_name+'.txt'
    edge_index = np.empty((2,edge_num*2))
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith('e'):
                parts = line.split()
                if len(parts) >= 3:
                    row = int(parts[1])
                    col = int(parts[2])
                    edge_index[0][count_edge]=row
                    edge_index[1][count_edge]=col
                    count_edge+=1
                    edge_index[0][count_edge] = col
                    edge_index[1][count_edge] = row
                    count_edge += 1
            elif line.startswith('v'):
                parts = line.split()
                if len(parts) >= 3:
                    count=int(parts[1])
                    feature_lable[count]=int(parts[2])
                    y[count]=int(parts[2])
                    # count+=1

    edge_index_torch = torch.from_numpy(edge_index).to(torch.long)
    return feature_lable,edge_index_torch
def graph_to_x(i):
    # amazon_table>2
    feature_lable_dict = {}
    start = datetime.now()
    print("开始：" + str(start))
    nhid = 32
    dropout = 0.5
    file_name = file_name_dict[i]
    v_num = v_num_dict[i]
    edge_num = edge_num_dict[i]
    nfeatuer = featuer_num_dict[i]

    # 检查CUDA是否可用，如果可用则使用第一个CUDA设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feature_lable, edge_index = load_x_data_graph(file_name, edge_num)
    x = torch.zeros(v_num, nfeatuer, )
    for j in range(v_num):
        x[i][feature_lable[i] - 1] = 1
    data = Data(x=x, edge_index=edge_index)
    print("数据读完了：" + str(datetime.now() - start))
    sage_time = datetime.now()
    # print(data)
    if gnn_model == 'SAGE':
        model = GraphSAGE(num_features=data.num_features, num_classes=nfeatuer)
    if (gnn_model == 'GAT'):
        model = GAT(num_features=data.num_features, num_classes=nfeatuer)
    if (gnn_model == 'GIN'):
        model = GIN(num_features=data.num_features, num_classes=nfeatuer, dropout=dropout)
    if (gnn_model == 'GCN'):
        model = GCN(in_channels=data.num_features, hidden_channels=128, num_layers=2, out_channels=data.num_features,
                    dropout=0.0)
    # model = GraphSAGE(num_features=data.num_features, num_classes=nfeatuer)
    # model = GAT(num_features=data.num_features, num_classes=nfeatuer)
    # model = GIN(num_features=data.num_features, num_classes=nfeatuer, dropout=dropout)
    output = model(data.x, data.edge_index)

    # save_x(output, file_name)
    end = datetime.now()
    print("结束：" + str(end))
    print("模型耗时：" + str(end - sage_time))
    print("---------------------")
    return output
def get_graph_x(file_name):
    x=np.zeros((v_num,nfeatuer),dtype=np.float64)
    with open("data/grapg_x/"+file_name, 'r') as file:
        for line in file:
            numlist=line.split()
            for i in range(v_num):
                for j in range(nfeatuer):
                    x[i][j]=numlist[j]

    return x

now = datetime.now()
file_name_dict =["mico"]
v_num_dict=[100000]
edge_num_dict=[1080298]
featuer_num_dict=[60]
ks=[100]
mode = ['EWS']

file_name=""
v_num =0
edge_num = 0
nfeatuer = 0
# ks=[300,400,500]

gnn_models = ["GAT"]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    for gnn_model in gnn_models:
        for k in ks:
            for i, file_name in enumerate(file_name_dict):
                motif_path = "Data/Motif/"+str(k)+"/" + str(file_name) + ".txt"
                data_path = "Data/" + file_name + ".txt"
                feature_lable_dict = {}
                start = datetime.now()

                # dropout = 0.5
                file_name = file_name_dict[i]
                v_num = v_num_dict[i]
                edge_num = edge_num_dict[i]
                nfeatuer = featuer_num_dict[i]

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                # device=torch.device('cpu')

                # ------------------------------------------------------
                x = torch.zeros(v_num, nfeatuer).to(device)
                y = torch.zeros(v_num, dtype=torch.long).to(device)
                feature_lable, edge_index = load_x_data_graph(file_name, edge_num, y)

                for ii in range(v_num):
                    if ii in feature_lable:
                        x[ii][feature_lable[ii]] = 1
                edge_index = edge_index.to(device)
                train_mask = torch.tensor(
                    [True] * v_num, dtype=torch.bool).to(device)
                test_mask = torch.tensor(
                    [True] * v_num, dtype=torch.bool).to(device)

                data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
                # ------------------------------------------------------------------------

                sage_time = datetime.now()
                if gnn_model == "GIN":

                    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    data.batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
                    model = GIN(in_channels=data.num_features, hidden_channels=128, out_channels=data.num_features,
                                num_layers=2, dropout=0.1).to(device)

                if gnn_model == "SAGE":
                    model = GraphSAGE(in_channels=data.num_features, hidden_channels=512,
                                      out_channels=data.num_features).to(device)
                if gnn_model == "GAT":
                    model = GAT(in_channels=data.num_features, hidden_channels=64, out_channels=data.num_features,
                                num_heads=1,
                                dropout=0.3).to(device)
                if gnn_model == "GCN":
                    model = GNN(data.num_features, 32, 0.0).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

                criterion = nn.CrossEntropyLoss()

                y = data.y.long()
                train_mask = data.train_mask.bool()

                if file_name!="twitch" and file_name!="patent":
                    for epoch in range(200):
                        model.train()
                        optimizer.zero_grad()
                        if gnn_model == 'GIN':
                            out = model(data.x, data.edge_index, data.batch)
                        else:
                            out = model(data.x, data.edge_index)
                        loss = criterion(out[data.train_mask], data.y[data.train_mask])
                        loss.backward()
                        optimizer.step()
                        scheduler.step(loss.item())
                        if (epoch + 1) % 10 == 0:
                            model.eval()
                            if gnn_model == 'GIN':
                                pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                            else:
                                pred = model(data.x, data.edge_index).argmax(dim=1)
                            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                            acc = int(correct) / int(data.test_mask.sum())
                            print(f'Epoch: {epoch + 1}, Accuracy: {acc:.4f}')
                            model.train()
                torch.cuda.empty_cache()
                if file_name!="twitch" and file_name!="patent":
                    output = model(data.x, data.edge_index)
                else:
                    output = get_grapg_x(file_name, gnn_model)

                start_time = time.time()
                # snapshot1 = tracemalloc.take_snapshot()
                wnode = get_wnode(data_path, motif_path, output, data,file_name)
                save_wnode(wnode, file_name, k, gnn_model)
                end_time = time.time()