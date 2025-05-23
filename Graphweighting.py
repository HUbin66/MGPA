import json

import networkx as nx
import numpy as np
import torch

from Pattern import Pattern
import torch.nn.functional as F
import Utils
from collections import defaultdict

from test import count_node_attributes


def get_weight_graph(data_path,out_graph_path,k,file_name,gnn_model):
    G = nx.Graph()

    label_counts = {}
    with open(data_path) as file:
        for line in file.readlines():
            data = line.split()
            # print(data)
            if data[0] == 'v':
                label = str(data[2])
                G.add_node(data[1], label=label, label_weight=0, node_weight=0)
                if label in label_counts:
                    label_counts[label]+=1
                else:
                    label_counts[label]=1
            elif data[0] == 'e':
                G.add_edge(data[1], data[2], label=str(data[3]), edge_weight=0)
    v_num =G.number_of_nodes()
    if not nx.is_connected(G):
        print('remove small connected components!')
        print('begin nodes:%d, edges:%d' % (G.number_of_nodes(), G.number_of_edges()))
        maxc = len(max(nx.connected_components(G), key=lambda x: len(x)))
        remove_list = []
        for c in nx.connected_components(G):
            if len(c) != maxc:
                remove_list += c
                for node in c:
                    label = G.nodes[node]['label']
                    label_counts[label] -= 1
        G.remove_nodes_from(remove_list)
        print('now nodes:%d, edges:%d' % (G.number_of_nodes(), G.number_of_edges()))


    Wnode = Utils.get_wnode(file_name,k,gnn_model)
    Wnode = Wnode.numpy()
    Wnode=normalize(Wnode)

    fb_hb = {}
    v_num_hb=0
    for i in range(len(Wnode)):
        if G.nodes.get(str(i)) is None:
            continue
        v_num_hb += 1
        w = Wnode[i]

        qz = round(w, 1)

        interval_key = str(qz * 1)

        if interval_key in fb_hb:
            fb_hb[interval_key] += 1
        else:
            fb_hb[interval_key] = 1

    sorted_fb_hb = dict(sorted(fb_hb.items()))
    percentages = {key: round ((value / v_num_hb),2)  for key, value in sorted_fb_hb.items()}
    print(percentages)
    top_10_indices = np.argsort(Wnode)[-10:][::-1]
    print(top_10_indices)


    for i in range(v_num) :
        if G.nodes.get(str(i)) is None:
            continue
        G.nodes[str(i)]['node_weight'] = Wnode[i]

    node_weights = [(node, G.nodes[node]['node_weight']) for node in G.nodes()]

    node_weights.sort(key=lambda x: x[1], reverse=True)

    return reset_index(G, out_graph_path)

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

def reset_index(G: nx.Graph,out_graph_path):
    newG = nx.Graph()
    index = {node: idx for idx, node in enumerate(G.nodes())}
    for in_node, out_node, attributes in G.edges(data=True):
        a1=G.nodes[in_node]['node_weight']
        a2=G.nodes[out_node]['node_weight']
        newG.add_node(index[in_node], label=G.nodes()[in_node]['label'],label_weight=G.nodes()[in_node]['label_weight'], node_weight=G.nodes()[in_node]['node_weight'])
        newG.add_node(index[out_node], label=G.nodes()[out_node]['label'],label_weight=G.nodes()[out_node]['label_weight'], node_weight=G.nodes()[out_node]['node_weight'])
        newG.add_edge(index[in_node], index[out_node], label=1, edge_weight=attributes['edge_weight'])
    print('reset index!')
    return newG

def get_single_edge(G: nx.Graph):
    sum_sup=0.0
    _dict = defaultdict(lambda: defaultdict(lambda: [set(), set()]))
    single_edge_sup=defaultdict(lambda: defaultdict(lambda: float))
    for edge in G.edges:
        _dict[G.nodes[edge[0]]['label']][G.nodes[edge[1]]['label']][0].add(edge[0])
        _dict[G.nodes[edge[0]]['label']][G.nodes[edge[1]]['label']][1].add(edge[1])
        _dict[G.nodes[edge[1]]['label']][G.nodes[edge[0]]['label']][0].add(edge[1])
        _dict[G.nodes[edge[1]]['label']][G.nodes[edge[0]]['label']][1].add(edge[0])

    for node1_label, node2 in _dict.items():
        for node2_label, ins in node2.items():
            if node2_label == node1_label:
                continue
            node1_ins, node2_ins = ins

            if len(node1_ins)<=len(node2_ins):
                single_edge_sup[node1_label][node2_label] = len(node1_ins)
                single_edge_sup[node2_label][node1_label] = len(node1_ins)
                sum_sup+=len(node1_ins)
            else:
                single_edge_sup[node1_label][node2_label] = len(node2_ins)
                single_edge_sup[node2_label][node1_label] = len(node2_ins)
                sum_sup += len(node2_ins)

    return single_edge_sup,sum_sup/2