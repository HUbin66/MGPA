import random
import json
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from torch.nn import Linear
from tqdm import tqdm

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from model.gat import GAT


# class NodeWeightModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(NodeWeightModel, self).__init__()
#         self.conv1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)
#         self.conv2 = GATConv(hidden_dim , hidden_dim, heads=2, concat=True)
#
#
#     def forward(self, x, edge_index):
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.elu(self.conv2(x, edge_index))
#         return x
def get_G(data_path):
    G = nx.Graph()
    label_counts = {}
    with open(data_path) as file:
        for line in file.readlines():
            data = line.split()
            if data[0] == 'v':
                label = str(data[2])
                G.add_node(data[1], label=label, label_weight=0.0, node_weight=0.0)
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            elif data[0] == 'e':
                G.add_edge(data[1], data[2], label=str(data[3]), edge_weight=0.0)

    return G

def linear_normalize(data, new_min, new_max):
    # 找到原始数据的最小值和最大值
    old_min = min(data)
    old_max = max(data)

    # 确保 old_min 不等于 old_max，避免除以零
    if old_min == old_max:
        return [new_min] * len(data)

    # 线性归一化
    normalized_data = [
        (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        for x in data
    ]

    return normalized_data
def get_motif_list(motif_path):

    motif_list = []
    with open(motif_path) as file:
        motif = {}
        for line in file.readlines():
            line = line.strip()
            if line.endswith(":"):
                if motif:
                    motif_list.append(motif)
                    motif = {}

                weight = int(line.split(":")[0])
                motif['weight'] = weight
            else:
                data = line.split()
                if data[0] == 'v':
                    node_id = int(data[1])
                    label = int(data[2])
                    motif.setdefault('nodes', {})[node_id] = label
                elif data[0] == 'e':
                    node1 = int(data[1])
                    node2 = int(data[2])

                    motif.setdefault('edges', {})[(node1, node2)] = None
        if motif:
            motif_list.append(motif)


    total_weight = sum(motif['weight'] for motif in motif_list)

    motif_weights = [motif['weight'] for motif in motif_list]



    for i,motif in enumerate(motif_list) :
        motif['normalized_weight'] = motif['weight'] / total_weight
        # motif['normalized_weight'] = motif_weights[i]

    return motif_list



def normalize(weights):
    min_val = 12000000
    max_val = -1200000
    for w in weights:
        if(w==0): continue
        if(w>max_val):max_val = w
        if(w<min_val):min_val = w
    for i,w in enumerate(weights):
        if(w==0): continue
        weights[i]=(w-min_val)/(max_val-min_val)
    return weights

# @profile
def get_wnode(data_path, motif_path,x,data,file_name):
    # TODO 稠密图开1：5
    motif_list = get_motif_list(motif_path)
    G = get_G(data_path)
    num_nodes = data.num_nodes
    num_motifs = len(motif_list)
    y_motif = np.zeros((num_nodes,len(x[0])))

    motif_features = np.zeros((num_nodes, num_motifs))
    node_attributes = np.zeros((num_nodes, 4))

    motif_match_count={}

    for node in tqdm(G.nodes()):
        neighbors = list(G.neighbors(node))
        subgraph_nodes = [node] + neighbors
        subgraph = G.subgraph(subgraph_nodes)

        for motif_idx, motif in enumerate(motif_list):

            motif_nodes = motif.get('nodes', {})
            flag=0
            for id, label in motif_nodes.items():
                if G.nodes[node]['label'] == str(label):
                    flag = 1
                    break
            if flag==0: continue;
            motif_edges = motif.get('edges', {})
            motif_weight = motif.get('normalized_weight')


            node_mapping = {}
            for motif_node, label in motif_nodes.items():

                for subgraph_node in subgraph_nodes:
                    if subgraph.nodes[subgraph_node].get('label', None) == str(label) and subgraph_node not in node_mapping.values():
                        node_mapping[motif_node] = subgraph_node
                        break

            if len(motif_nodes) == 2:

                for motif_edge in motif_edges:
                    u, v = motif_edge
                    if u not in node_mapping or v not in node_mapping:
                        continue
                    subgraph_edge_1 = (node_mapping[u], node_mapping[v])
                    subgraph_edge_2 = (node_mapping[v], node_mapping[u])
                    if not (subgraph.has_edge(*subgraph_edge_1) or subgraph.has_edge(*subgraph_edge_2)):
                        continue
                    else:
                        if node in node_mapping.values():

                            if node!=node_mapping[u] and node!=node_mapping[v]: continue;
                            motif_features[int(node), motif_idx] = motif_weight

                            y_motif[int(node), int(G.nodes[node_mapping[u]].get("label"))] += motif_weight*(2/len(motif_nodes))
                            y_motif[int(node), int(G.nodes[node_mapping[v]].get("label"))] += motif_weight*(2/len(motif_nodes))
                            motif_match_count[node]=motif_match_count.get(node, 0) + 1


    node_weights= np.zeros((num_nodes))

    if file_name!="twitch" and file_name!="patent":
        x = x.detach().cpu().numpy()

    for i in range(len(x)):
        x[i] = normalize(x[i])
    for i,y in enumerate(y_motif):
        sum=0
        for j in range(len(y)):
            label_j=str(j)
            if G.nodes.get(str(i)) is None:
                continue
            neis=G.neighbors(str(i))
            cnt=0
            for node in neis:
                if G.nodes[node]['label'] == label_j: cnt+=1
            if cnt!=0:
                sum+= y[j]*(x[i][j]/cnt)
            if cnt==0:
                sum += y[j] * (x[i][j])

        if file_name=='dblp' and  str(i) not in G:
            continue
        if len(list(G.neighbors(str(i))))!=0:
            sum=sum*(motif_match_count.get(str(i),0)/len(list(G.neighbors(str(i)))))
        else: sum=0
        node_weights[i]=sum

    return node_weights