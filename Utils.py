import os
import networkx
import networkx as nx
import numpy as np
import pandas as pd
import time
import csv



from torch_geometric.nn import GCN
import numpy as np
import torch
import torch
from torch_geometric.data import Data


from model.gin import GIN
from model.sage import GraphSAGE


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
def save_x(output,file_name,gnn_model):
    output_numpy = output.detach().cpu().numpy()

    with open('Data/output_x/'+gnn_model+'/'+file_name+'.txt', 'w') as file:
        for row in output_numpy:

            row_str = ' '.join(map(str, row))

            file.write(row_str + '\n')

    output_numpy = output.detach().cpu().numpy()

def save_gat(output,file_name,gnn_model):

    output_numpy = output.detach().cpu().numpy()


    with open('get_test.txt', 'w') as file:

        for row in output_numpy:

            row_str = ' '.join(map(str, row))

            file.write(row_str + '\n')

    output_numpy = output.detach().cpu().numpy()



def graph_to_x(graph,nfeatuer,gnn_model):
    x=[]
    vnode=0
    edge_num=0

    for line in graph:
        if (line[0] == 'v'):
            vnode += 1
        else: edge_num +=1
    # adj = torch.zeros((vnode, vnode), dtype=torch.float, device=device)
    count_edge = 0
    edge_index = np.empty((2, edge_num * 2))
    for line in graph:
        if(line[0]=='e'):
            parts = line.split()
            row = int(parts[1])
            col = int(parts[2])
            edge_index[0][count_edge] = row
            edge_index[1][count_edge] = col
            count_edge += 1
            edge_index[0][count_edge] = col
            edge_index[1][count_edge] = row
            count_edge += 1

    # x_feat=torch.zeros((vnode,25),dtype=torch.float, device=device)
    x_feat = torch.zeros((vnode, nfeatuer), dtype=torch.float)
    for i in range(vnode):
        xx=graph[i]
        xx=xx.split(" ")
        x_feat[i][int(xx[2])]=1

    return x_feat

def com_top_K_patter_x(file_name,nfeatuer,gnn_model):
    x_fre_pattern=[]

    # with open("Data/pattern_graph/old_graph/"+file_name+"/200.lg") as file:
    with open("Data/Motif/500/"+file_name+'.txt') as file:
        current_block = []
        for line in file:
            line = line.strip()
            if line.startswith('v') or line.startswith('e'):

                current_block.append(line)


            elif ':' in line:

                if current_block:
                    x_part=graph_to_x(current_block,nfeatuer,gnn_model)
                    x_fre_pattern.append(x_part)
                    current_block = []


        if current_block:
            x_part=graph_to_x(current_block,nfeatuer,gnn_model)
            x_fre_pattern.append(x_part)

        # x_fre_pattern = torch.cat(x_fre_pattern, dim=0)
        return x_fre_pattern

def sup_pattern(file_name,k):
    numlist=np.zeros(k)
    supmin=np.zeros(k)
    # with open("data/MNI/" + file_name+".lg", 'r') as file:
    with open("Data/MNI/Motif/500/" + file_name + ".txt", 'r') as file:
        sum=0
        js=0;
        for line in file:
            if(js>=k): break
            numlist[js]=int(line)
            sum+=int(line)
            js+=1
        for i in range(js):
            # supmin[i]=numlist[i]/sum
            supmin[i] = numlist[i]

    return supmin


def save_wnode(wnode,file_name,k,gnn_model):

    file_path="Data/wnode/"+gnn_model+"/"+file_name+str(k)+'.txt'

    np.savetxt(file_path, wnode)


def get_wnode():

    file_path = 'wnode_data2.txt'

    Wnode_numpy = np.loadtxt(file_path)

    Wnode = torch.tensor(Wnode_numpy, dtype=torch.float64)
    return Wnode


def save_graph_with_weights_agg(G, outpath):
    node_weights = [(node, G.nodes[node]['node_weight']) for node in G.nodes()]

    node_weights.sort(key=lambda x: x[1], reverse=True)

    top_10_nodes = node_weights[:10]
    new_output_path = '/'.join(outpath.split('/')[:-1])
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path)
    with open(outpath, 'w') as file:
        file.writelines('# t l || v id label node_weight || e id1 id2 edge_weight \n')
        num = 0
        d = {}
        for node in G.nodes:
            file.writelines('v %s %s %.4f\n' % (num, G.nodes[node]['label'], G.nodes[node]['node_weight']))
            d[node] = num
            num += 1
        for edge in G.edges:
            file.writelines('e %d %d %.2f\n' % (d[edge[0]], d[edge[1]], G.edges[edge]['edge_weight']))
        file.close()

def get_graph_from_path(data_path):
    G = nx.Graph()


    with open(data_path) as file:
        for line in file.readlines():
            data = line.split()
            if data[0] == 'v':
                G.add_node(data[1], label=str(data[2]))
            elif data[0] == 'e':
                G.add_edge(data[1], data[2], label=str(data[3]))
    # print(G.number_of_nodes())
    if not nx.is_connected(G):
        start_time = time.time()

        print('remove small connected components!')
        print('begin nodes:%d, edges:%d'% (G.number_of_nodes(),G.number_of_edges()))
        maxc = len(max(nx.connected_components(G), key=lambda x: len(x)))
        remove_list = []
        for c in nx.connected_components(G):
            if len(c) != maxc:
                remove_list += c
        G.remove_nodes_from(remove_list)
        print('now nodes:%d, edges:%d'%(G.number_of_nodes(),G.number_of_edges()))
        end_time = time.time()
        use_time = end_time - start_time

        df = pd.DataFrame({'use_time': [use_time], 'data_path': [data_path],
                           'time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]})
        df.to_csv('./log/exp/_del_connected_time.csv', index=False, mode='a', header=False)
    return reset_index(G)

def reset_index(G:networkx.Graph()):
    newG = networkx.Graph()
    index = {node:idx for idx,node in enumerate(G.nodes())}
    for in_node, out_node in G.edges():
        newG.add_node(index[in_node], label=G.nodes()[in_node]['label'])
        newG.add_node(index[out_node], label=G.nodes()[out_node]['label'])
        newG.add_edge(index[in_node],index[out_node])
    print('reset index!')
    return newG

def get_weighted_graph(data_path):
    G = nx.Graph()
    nodes=[]
    edges=[]
    with open(data_path) as file:
        for line in file.readlines():
            data = line.split()
            if data[0] == 'v':
                node = (data[1], {'label': str(data[2]), 'node_weight': float(data[3])})
                nodes.append(node)
                # G.add_node(data[1], label=str(data[2]), weight=float(data[3]))
            elif data[0] == 'e':
                edge = (data[1], data[2], {'label': str(data[3]), 'edge_weight': float(data[3])})
                edges.append(edge)
                # G.add_edge(data[1], data[2], label=str(data[3]), weight=float(data[3]))

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if not nx.is_connected(G):
        start_time = time.time()

        print('remove small connected components!')
        print('begin nodes:%d, edges:%d' % (G.number_of_nodes(), G.number_of_edges()))
        maxc = len(max(nx.connected_components(G), key=lambda x: len(x)))
        remove_list = []
        for c in nx.connected_components(G):
            if len(c) != maxc:
                remove_list += c
        G.remove_nodes_from(remove_list)

        print('now nodes:%d, edges:%d' % (G.number_of_nodes(), G.number_of_edges()))
        end_time = time.time()
        use_time = end_time - start_time



    return reset_index_with_weight(G)

def reset_index_with_weight(G: nx.Graph):
    newG = nx.Graph()
    index = {node: idx for idx, node in enumerate(G.nodes())}
    for in_node, out_node, attributes in G.edges(data=True):
        newG.add_node(index[in_node], label=G.nodes()[in_node]['label'], node_weight=G.nodes()[in_node]['node_weight'])
        newG.add_node(index[out_node], label=G.nodes()[out_node]['label'],node_weight=G.nodes()[out_node]['node_weight'])
        newG.add_edge(index[in_node], index[out_node], label=1, edge_weight=attributes['edge_weight'])

    print('reset index!')
    return newG

def get_wnode(file_name,k,gnn_model):

    file_path = 'Data/wnode/'+gnn_model+'/'+file_name+str(k)+'.txt'

    Wnode_numpy = np.loadtxt(file_path)

    Wnode = torch.tensor(Wnode_numpy, dtype=torch.float64)
    return Wnode

def save_motif(motif_list:list, out_path):
    # motif_list=sorted()
    new_output_path = '/'.join(out_path.split('/')[:-1])
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path)
    # file = open(out_path, 'w')
    with open(out_path, 'w') as file:
        # print(file)
        for motif in motif_list:
            num = 0
            d = {}
            print(motif.get_mni() )
            file.writelines(f'{motif.get_mni()}:\n')
            for node in motif.nodes:
                file.writelines('v %s %s \n' % (num, motif.nodes[node]['label']))
                d[node] = num
                num += 1
            for edge in motif.edges:
                file.writelines('e %d %d 1\n' % (d[edge[0]], d[edge[1]]))

def save_graph_with_weights(G, outpath):
    new_output_path = '/'.join(outpath.split('/')[:-1])
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path)
    with open(outpath, 'w') as file:
        file.writelines('# t l\n')
        num = 0
        d = {}
        for node in G.nodes:
            file.writelines('v %s %s %.2f\n' % (num, G.nodes[node]['label'],G.nodes[node]['weight']))
            d[node] = num
            num += 1
        for edge in G.edges:
            file.writelines('e %d %d 1\n' % (d[edge[0]], d[edge[1]]))
        file.close()

def get_grapg_x(file_name,gnn_model):

    file_path = 'Data/output_x/'+gnn_model+'/' +file_name + '.txt'

    feature_vectors = []


    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:

                vector = line.strip().split()

                vector = [float(value) for value in vector]

                feature_vectors.append(vector)
    except FileNotFoundError:
        print("ERROR")
    except ValueError:
        print("ERROR")
    except Exception as e:
        print("ERROR")

    return feature_vectors
