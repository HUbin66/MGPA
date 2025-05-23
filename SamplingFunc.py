import math
import random

import networkx as nx
#from littleballoffur import HybridNodeEdgeSampler



def random_edge(G,size):
    G1 = nx.Graph()
    edge_nums = int(len(G.nodes()) * size)
    edges_sampling = random.sample(G.edges(), edge_nums)
    for node1, node2 in edges_sampling:
        G1.add_node(node1, label=G.nodes()[node1]['label'])
        G1.add_node(node2, label=G.nodes()[node2]['label'])
        G1.add_edge(node1, node2)
        node_set = set(G1.nodes())
        for x in (node1,node2):
            neigh = (node_set & set(list(G.neighbors(x))))
            for y in neigh:
                G1.add_edge(x, y)
        if len(G1.edges()) >= edge_nums:
            break
    return G1


def random_edge1(G,size):
    G1 = nx.Graph()
    # edge_nums = int(len(G.edges()) * size)
    node_nums = int(len(G.nodes) * size)
    edges_sampling = random.sample(G.edges(), len(G.edges))
    for node1, node2 in edges_sampling:
        G1.add_node(node1, label=G.nodes()[node1]['label'])
        G1.add_node(node2, label=G.nodes()[node2]['label'])
        G1.add_edge(node1, node2)
        node_set = set(G1.nodes())
        for x in (node1,node2):
            neigh = (node_set & set(list(G.neighbors(x))))
            for y in neigh:
                G1.add_edge(x, y)
        if len(G1.nodes()) >= node_nums:
            break
    return G1

def random_edge_(G,size):
    G1 = nx.Graph()
    # edge_nums = int(len(G.edges()) * size)
    edge_nums = int(len(G.nodes()) * size)
    edges_sampling = random.sample(G.edges(), edge_nums)
    for node1, node2 in edges_sampling:
        G1.add_node(node1, label=G.nodes()[node1]['label'])
        G1.add_node(node2, label=G.nodes()[node2]['label'])
        G1.add_edge(node1, node2)
    node_set = set(G1.nodes())
    for x in G1.nodes():
        neigh = (node_set & set(list(G.neighbors(x))))
        for y in neigh:
            G1.add_edge(x, y)
    print("nodes",len(G1.nodes()),', edges:',len(G1.edges))
    return G1

def random_node(G,size):
    node_nums = int(len(G.nodes()) * size)
    nodes_sampling = random.sample(G.nodes(), math.floor(node_nums))
    G1 = nx.Graph()
    for node in nodes_sampling:
        G1.add_node(node,label=G.nodes()[node]['label'])

    node_set = set(G1.nodes())
    for x in G1.nodes():
        neigh = (node_set & set(list(G.neighbors(x))))
        for y in neigh:
            G1.add_edge(x, y)
    return G1



