import os
import random

# import igraph
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

import SamplingFunc
import Utils
import littleballoffur
from littleballoffur import HybridNodeEdgeSampler
from Utils import get_graph_from_path

def draw(G):
    pos = nx.spring_layout(G, iterations=20)
    nx.draw(G, pos)
    node_labels = nx.get_node_attributes(G, 'label')
    print(node_labels)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=20)
    plt.show()


def addLabel(G,newG):
    print(newG.number_of_nodes())
    print(G.nodes[100])
    labels = G.nodes()
    for id in newG.nodes:
        print(id)

    print(newG.number_of_nodes())

def save_graph(G, outpath):
    new_output_path = '/'.join(outpath.split('/')[:-1])
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path)
    file = open(outpath, 'w')
    file.writelines('# t l\n')
    num = 0
    d = {}

    for node in G.nodes:

        node_attrs = G.nodes[node]

        label = node_attrs.get('label', None)


        file.writelines(f'v {num} {label}\n')
        d[node] = num
        num += 1

    for edge in G.edges:
        u, v = edge

        if u not in d or v not in d:
            continue

        file.writelines(f'e {d[u]} {d[v]} 1\n')
    file.close()

def by_random_edge(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):
        seed=random.randint(1, 10000)
        print(seed)
        sampler = littleballoffur.RandomEdgeSampler(int(G.number_of_edges()*i),int(G.number_of_nodes()*i),seed=seed)
        newG = sampler.sample(G)
        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))
        print("random edge %s-%.3f Done!"%(filename,i))


def by_hybird_nodeedge(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):
        sampler = littleballoffur.HybridNodeEdgeSampler(int(G.number_of_edges()*i))
        newG = sampler.sample(G)
        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))
        print("random edge %s-%.3f Done!"%(filename,i))

def by_random_walk(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):

        sampler = littleballoffur.RandomWalkSampler(int(G.number_of_nodes()*i))
        newG = sampler.sample(G)

        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))
        print("random walk %s-%.3f Done!" % (filename, i))



def by_forest_fire(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):

        seed = random.randint(1, 10000)
        print(seed)
        sampler = littleballoffur.ForestFireSampler(int(G.number_of_nodes()*i),seed=seed)
        newG = sampler.sample(G)

        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))

# spikyball
def by_spikyball(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):

        seed = random.randint(1, 10000)

        sampler = littleballoffur.SpikyBallSampler(int(G.number_of_nodes()*i),seed=seed)
        newG = sampler.sample(G)
        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))


def by_common_neighbor_aware_random_walk(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):
        seed = random.randint(1, 10000)
        print(seed)
        sampler = littleballoffur.CommonNeighborAwareRandomWalkSampler(int(G.number_of_nodes() * i),seed=seed)
        newG = sampler.sample(G)
        # addLabel(G,newG)
        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))

def by_random_node(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):

        seed = random.randint(1, 10000)

        sampler = littleballoffur.RandomNodeSampler(int(G.number_of_nodes()*i),seed=seed)
        newG = sampler.sample(G)
        # addLabel(G,newG)
        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))


# random node edge
# @profile(precision=4, stream=open('log/exp/20221201_mem.log', 'w+'))
def by_random_node_edge(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):
        # newG = Graph_Sampling.TIES().ties(G, 0, int(i * 100))
        newG = SamplingFunc.random_edge_(G,i)
        # sampler = littleballoffur.RandomNodeEdgeSampler(G.number_of_edges()*i)
        # newG = sampler.sample(G)
        # addLabel(G,newG)
        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))
        print("random node %s-%.3f Done!" % (filename, i))

# @profile(precision=4, stream=open('log/exp/20221201_mem.log', 'w+'))
def by_hybrid_node_edge(filename,data_path, out_path, start, end, interval):
    G = get_graph_from_path(data_path)
    for i in np.arange(start, end, interval):
        number_of_edges = int(i * G.number_of_edges())
        sampler= HybridNodeEdgeSampler(number_of_edges=number_of_edges)
        # t = [i for i in range(G.number_of_nodes())]
        # l = sorted([node for node in G.nodes()])
        # print(t[-1],l)
        newG = sampler.sample(G)
        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))
        print("hybrid node edge %s-%.3f Done!" % (filename, i))


# @profile(precision=4, stream=open('../log/exp/_mem.log', 'w+'))
def by_random_weight_node(filename,data_path, out_path, start, end, interval):
    G = Utils.get_graph_with_weights(data_path)
    for i in np.arange(start, end, interval):
        sampler = littleballoffur.WeightsBasedSampler(int(G.number_of_nodes() * i))
        newG = sampler.sample(G)
        save_graph(newG, '%s/%s-%.3f.lg' % (out_path, filename, i))
        print("weight node %s-%.3f Done!" % (filename, i))


def by_random_walk_aggregate(filename,data_path, out_path, start, end, interval):
    # print(data_path)
    G = Utils.get_graph_with_weights_agg(data_path)
    for i in np.arange(start, end, interval):
        sampler = littleballoffur.RandomWalkBasedAggregateSampler(int(G.number_of_nodes() * i))
        newG=sampler.sample(G)
        save_graph(newG,'%s/%s-%.3f.lg' % (out_path, filename, i))
        print("random walk aggregate %s-%.3f Done!" % (filename, i))


def by_random_walk_weight(filename,data_path, out_path, start, end, interval):
    G = Utils.get_graph_with_weights(data_path)
    for i in np.arange(start, end, interval):
        sampler = littleballoffur.RandomWalkBasedWeightsSampler(int(G.number_of_nodes() * i))
        newG=sampler.sample(G)
        save_graph(newG,'%s/%s-%.3f.lg' % (out_path, filename, i))
        print("random walk weight %s-%.3f Done!" % (filename, i))






#统一
def by_edge_weight_sampling(filename,data_path, out_path, start, end, interval,k,G):


    label_penalty = {}
    if filename =='dblp' or filename =='patent' or filename=='twitch':
        back = False
    else : back = True
    yu=True
    for i in np.arange(start, end, interval):
        seed = random.randint(1, 10000)

        if filename =='mico' or filename =='twitter':
            sampler = littleballoffur.EdgeWeightSampler_mico(int(G.number_of_nodes() * i), seed, label_penalty, back)
        else :
            sampler = littleballoffur.EdgeWeightSampler_patent(int(G.number_of_nodes() * i), seed, label_penalty, back)
        newG = sampler.sample(G)
        xx=out_path+'/'+str(k)+'/'+filename+'_'+str(i)+'.lg'
        save_graph(newG,xx)
        # save_graph(newG, out_path)
        print("random walk aggregate %s-%.3f Done!" % (filename, i))