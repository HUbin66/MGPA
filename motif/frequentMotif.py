import copy
import random
from collections import defaultdict
import time
from queue import PriorityQueue

import pandas as pd
from networkx.algorithms import isomorphism as iso


import networkx as nx
import itertools as it

import Utils
from Pattern import Pattern

from motif import GlobalVar


# 从data_path载入图,初始化weight=1
def read_graph_with_default_weights(data_path):
    G = nx.Graph()
    with open(data_path) as file:
        for line in file.readlines():
            data = line.split()
            if data[0] == 'v':
                G.add_node(data[1], label=str(data[2]), weight=1.0)
            elif data[0] == 'e':
                G.add_edge(data[1], data[2], label=str(data[3]), weight=1.0)
    if not nx.is_connected(G):
        print('remove small connected components!')
        print('begin nodes:%d, edges:%d' % (G.number_of_nodes(), G.number_of_edges()))
        maxc = len(max(nx.connected_components(G), key=lambda x: len(x)))
        remove_list = []
        for c in nx.connected_components(G):
            if len(c) != maxc:
                remove_list += c
        G.remove_nodes_from(remove_list)
        print('now nodes:%d, edges:%d' % (G.number_of_nodes(), G.number_of_edges()))
    return reset_index(G)


# 删除小的联通子图以后重新设置index
def reset_index(G: nx.Graph):
    newG = nx.Graph()
    index = {node: idx for idx, node in enumerate(G.nodes())}
    for in_node, out_node in G.edges():
        newG.add_node(index[in_node], label=G.nodes()[in_node]['label'], weight=1.0)
        newG.add_node(index[out_node], label=G.nodes()[out_node]['label'], weight=1.0)
        newG.add_edge(index[in_node], index[out_node])
    print('reset index!')
    return newG


# 获取所有的单边
def get_single_edge(G: nx.Graph):
    # _dict1 = defaultdict(lambda: defaultdict(lambda: [set(), set()]))
    _dict = defaultdict(lambda: defaultdict(lambda: [set(), set()]))
    for edge in G.edges:
        _dict[G.nodes[edge[0]]['label']][G.nodes[edge[1]]['label']][0].add(edge[0])
        _dict[G.nodes[edge[0]]['label']][G.nodes[edge[1]]['label']][1].add(edge[1])
        _dict[G.nodes[edge[1]]['label']][G.nodes[edge[0]]['label']][0].add(edge[1])
        _dict[G.nodes[edge[1]]['label']][G.nodes[edge[0]]['label']][1].add(edge[0])

    # count = 0
    # for key1, sub_dict in _dict1.items():
    #     if count >= 88:
    #         break
    #     for key2, value in sub_dict.items():
    #         _dict[key1][key2] = value  # 将值直接复制到 _dict
    #     count += 1

    # print(_dict.keys())
    single_edge_list = []
    single_edge_index = defaultdict(lambda: {})
    for node1_label, node2 in _dict.items():
        for node2_label, ins in node2.items():
            if node2_label == node1_label:
                continue
            node1_ins, node2_ins = ins
            single_edge = Pattern()
            single_edge.add_node(node1_label, ins=node1_ins)
            single_edge.add_node(node2_label, ins=node2_ins)
            single_edge.add_edge(node1_label, node2_label)
            # 防止出现相同的单边
            if node1_label in single_edge_index and node2_label in single_edge_index[node1_label]:
                continue
            single_edge_index[node1_label][node2_label] = single_edge
            single_edge_index[node2_label][node1_label] = single_edge
            single_edge_list.append(single_edge)

    # 按照MNI排序
    single_edge_list.sort(reverse=True, key=lambda edge: min([len(node[1]) for node in edge.nodes(data="ins")]))
    # single_edge_list.sort(reverse=True, key=lambda edge: get_MNI(edge))
    return single_edge_list, single_edge_index


def get_MNI(pattern: nx.Graph, single_edge_index=None):
    if 'ins' not in pattern.node_attr_dict_factory():
        ans = float('inf')
        for in_node, out_node in pattern.edges:
            in_label = pattern.nodes[in_node]['label']
            out_label = pattern.nodes[out_node]['label']
            ans = min(ans, min([len(ins) for ins in single_edge_index[in_label][out_label].nodes(data='ins')]))
        return ans
    return min([len(pattern.nodes[node]["ins"]) for node in pattern.nodes])


# 向前更新node的match
def pre_updata(G: nx.Graph, pattern: nx.Graph, single_edge_index: defaultdict, vis_edge: list, core_node: list):
    for edge, n in zip(vis_edge[:-1], core_node):
        node = pattern.nodes[n]
        all_neighbor = set()
        for _ in node['ins']:
            all_neighbor |= set(nx.neighbors(G, _))
        node = pattern.nodes[edge[1] if edge[0] == n else edge[0]]
        node['ins'] = node['ins'] & all_neighbor
        if len(node['ins']) < GlobalVar.MNI:
            return False
    return True


# 向后更新node的match
def nex_update(G, node, nex_node):
    all_neighbor = set()
    for _ in node['ins']:
        all_neighbor |= set(nx.neighbors(G, _))
    nex_node['ins'] = nex_node['ins'] & all_neighbor


# 获取整个pattern的match
def match_ins(G: nx.Graph, pattern: nx.Graph, single_edge_index: defaultdict):
    bfs = nx.bfs_edges(pattern, max(pattern.nodes.keys(), key=lambda node: pattern.degree(node)))
    pre_edge = None

    vis_edge = []
    vis_node = set()
    core_node = []

    for edge in bfs:
        vis_edge.append(edge)
        vis_node.add(edge[0])
        vis_node.add(edge[1])

        if pre_edge is None:
            pre_edge = edge
            continue
        core_node.append(edge[0] if edge[0] in pre_edge else edge[1])
        node, nex_node = (pattern.nodes[edge[0]], pattern.nodes[edge[1]]) if edge[0] in pre_edge else \
            (pattern.nodes[edge[1]], pattern.nodes[edge[0]])
        node_label = node['label']
        nex_label = nex_node['label']
        pre_node = pattern.nodes[pre_edge[0] if pre_edge[0] != core_node[-1] else pre_edge[1]]
        pre_label = pre_node['label']
        if node_label in single_edge_index[pre_label] and nex_label in single_edge_index[node_label]:
            node['ins'] = single_edge_index[pre_label][node_label].nodes[node_label]['ins'] & \
                          single_edge_index[node_label][nex_label].nodes[node_label]['ins'] & \
                          node['ins']
        else:
            return False
        # except:
        #     print(pre_label,node_label,nex_label)
        #     for edge in G.edges:
        #         if (edge[0]==node_label and edge[1]==nex_label) or (edge[1]==node_label and edge[0]==nex_label):
        #             print('ok')
        #     print(single_edge_index[node_label][nex_label])
        #     print(single_edge_index[node_label][nex_label].nodes())
        if pre_updata(G, pattern, single_edge_index, vis_edge, core_node):
            nex_update(G, node, nex_node)
        else:
            return False


# 检测同构
def check_iso(patterns, pattern):
    cnm = iso.categorical_node_match('label', -1)
    for p in patterns:
        if iso.GraphMatcher(pattern, p, cnm).is_isomorphic():
            return True
    return False


# 匹配motif
def match_motif(motif: nx.Graph, G: nx.Graph, single_edge_list: list, single_edge_index: defaultdict):
    index = list(single_edge_index.keys())
    index.sort(reverse=True,
               key=lambda l: sum([single_edge.get_mni() for single_edge in single_edge_index[l].values()]))
    # label_list = [label for label in it.permutations(index, motif.number_of_nodes())]
    label_list=[]
    label_generator = it.permutations(index, motif.number_of_nodes())
    js=0
    for label in label_generator:
        label_list.append(label)
        js+=1

    patterns = []
    pq = PriorityQueue()
    for labels in label_list:

        pattern = Pattern(motif.copy())

        # 附着标签
        for i, node in enumerate(pattern.nodes):
            pattern.nodes[node]["label"] = labels[i]

        # 附着单边ins
        flag = False
        for in_node, out_node in pattern.edges:
            in_label = pattern.nodes[in_node]['label']
            out_label = pattern.nodes[out_node]['label']
            if out_label in single_edge_index[in_label]:
                single_edge = single_edge_index[in_label][out_label]

                if 'ins' not in pattern.nodes[in_node]:
                    pattern.nodes[in_node]['ins'] = single_edge.nodes[in_label]['ins']
                else:
                    pattern.nodes[in_node]['ins'] = pattern.nodes[in_node]['ins'] & single_edge.nodes[in_label]['ins']
                    # print([len(pattern.nodes[in_node]['ins']), len(single_edge.nodes[in_label]['ins'])],GlobalVar.MNI)

                if 'ins' not in pattern.nodes[out_node]:
                    pattern.nodes[out_node]['ins'] = single_edge.nodes[out_label]['ins']
                else:
                    # print(len(pattern.nodes[out_label]['ins']), len(pattern.nodes[out_node]['ins']))
                    pattern.nodes[out_node]['ins'] = single_edge.nodes[out_label]['ins'] & pattern.nodes[out_node][
                        'ins']

                if min(len(pattern.nodes[in_node]['ins']), len(pattern.nodes[out_node]['ins'])) < GlobalVar.MNI:
                    flag = True
                    break
            else:
                flag = True
                break
        if flag:
            del pattern
            continue

        # 维护优先队列
        if pq.qsize() >= 33:
            p = pq.get()
            if p < pattern:
                # 更新实例
                if check_iso(patterns, pattern):
                    pq.put(p)
                    del pattern
                    continue
                match_ins(G, pattern, single_edge_index)
                if p < pattern:
                    del p
                    pq.put(pattern)
                    p = pq.get()
                    GlobalVar.MNI = p.get_mni()
                    pq.put(p)
                else:
                    pq.put(p)
            else:
                pq.put(p)
                continue
        else:
            # 更新实例
            if check_iso(patterns, pattern):
                continue
            match_ins(G, pattern, single_edge_index)
            pq.put(pattern)

        # print(sorted([pattern.get_mni() for pattern in pq.queue]))
        patterns.append(pattern)
    print("MNI:", GlobalVar.MNI)
    GlobalVar.MNI = 1
    print(len(patterns))
    return sorted(pq.queue, reverse=True)


def init_ins(motif, single_edge_index):
    for in_node, out_node in motif.edges:
        in_label = motif.nodes[in_node]['label']
        out_label = motif.nodes[out_node]['label']
        if out_label in single_edge_index[in_label]:
            single_edge = single_edge_index[in_label][out_label]
            if 'ins' not in motif.nodes[in_node]:
                motif.nodes[in_node]['ins'] = single_edge.nodes[in_label]['ins']
            else:
                motif.nodes[in_node]['ins'] =  single_edge.nodes[in_label]['ins'] & motif.nodes[in_node]['ins']

            if 'ins' not in motif.nodes[out_node]:
                motif.nodes[out_node]['ins'] = single_edge.nodes[out_label]['ins']
            else:
                motif.nodes[out_node]['ins'] = single_edge.nodes[out_label]['ins'] & motif.nodes[out_node]['ins']
        else:
            motif.nodes[in_node]['ins'] = set()
            motif.nodes[out_node]['ins'] = set()


def match_pattern(graph_path, motif_path,out_path):
    G = Utils.get_graph_from_path(graph_path)
    motif_list = Utils.get_motif_list_from_path(motif_path)
    single_edge_list, single_edge_index = get_single_edge(G)
    GlobalVar.MNI = 1
    for motif in motif_list:
        init_ins(motif, single_edge_index)
        match_ins(G, motif, single_edge_index)
    Utils.save_motif(motif_list,out_path)
    print('save match motif to',out_path)

# data_name = 'mico'
# match_pattern(f'../Data/{data_name}.txt',f'{data_name}_motifs.txt')

def get_motif1():
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_node(4)
    motif.add_edge(1, 2)
    motif.add_edge(3, 2)
    motif.add_edge(4, 2)
    return motif


def get_motif2():
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_node(4)
    motif.add_edge(1, 2)
    motif.add_edge(2, 3)
    motif.add_edge(3, 4)
    return motif


def get_motif3():
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_edge(1, 2)
    motif.add_edge(2, 3)
    return motif


def get_motif2v1b_1():#TODO:1-2-3是否等于1-3-2
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_edge(1, 2)
    return motif

def get_motif3v2b_1():#TODO:1-2-3是否等于1-3-2
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_edge(1, 2)
    motif.add_edge(2, 3)
    return motif

def get_motif3v2b_2():#TODO:1-2-3是否等于1-3-2
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_edge(1, 2)
    motif.add_edge(1, 3)
    return motif

def get_motif4v3b_1():
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_node(4)
    motif.add_edge(1, 2)
    motif.add_edge(1, 3)
    motif.add_edge(1, 4)
    return motif

def get_motif4v3b_2():
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_node(4)
    motif.add_edge(1, 2)
    motif.add_edge(2, 3)
    motif.add_edge(3, 4)
    return motif

def get_motif4v3b_3():
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_node(4)
    motif.add_edge(1, 2)
    motif.add_edge(1, 4)
    motif.add_edge(2, 3)
    return motif

def get_motif4v3b_4():
    motif = nx.Graph()
    motif.add_node(1)
    motif.add_node(2)
    motif.add_node(3)
    motif.add_node(4)
    motif.add_edge(1, 2)
    motif.add_edge(1, 4)
    motif.add_edge(3, 4)
    return motif

def get_motif_info(graph_path,out_graph_path,out_motif_path):
    G = read_graph_with_default_weights(graph_path)
    single_edge_list, single_edge_index = get_single_edge(G)
    motif_list = []
    # for i,motif in enumerate([get_motif1(),get_motif2(),get_motif3()]):
    # for i, motif in enumerate([get_motif2v1b_1(),get_motif3v2b_1(),get_motif4v3b_2(), get_motif4v3b_1(),get_motif4v3b_4(),get_motif4v3b_3()]):
    # for i, motif in enumerate([get_motif2v1b_1()):
    for i, motif in enumerate([get_motif2v1b_1(), get_motif3v2b_1(), get_motif3v2b_2()]):
        start_time = time.time()

        motif_list += match_motif(motif, G, single_edge_list, single_edge_index)

        end_time = time.time()
        use_time = end_time - start_time

        df = pd.DataFrame({'real_mining_time': [use_time], 'graph': [graph_path],'motif':[i],
                           'time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]})
        # df.to_csv('../log/exp/_motif_search_time.csv', index=False, mode='a', header=False)
        df.to_csv('log/exp/_motif_search_time.csv', index=False, mode='a', header=False)

    for pattern in motif_list:
        for node in pattern.nodes:
            for ins in pattern.nodes[node]['ins']:
                G.nodes[ins]['weight'] += 1
    Utils.save_graph_with_weights(G, out_graph_path)
    Utils.save_motif(motif_list,out_motif_path)



# data_name = 'mico'
# G = read_graph_with_default_weights("../Data/%s.txt" % data_name)
# # G = read_graph_with_default_weights("../Data/mico.txt")
# single_edge_list, single_edge_index = get_single_edge(G)
# motif_list = []
# start_t = time.perf_counter()
# motifs = match_motif(get_motif1(), G, single_edge_list, single_edge_index)
# motif_list += motifs
# dis = defaultdict(int)
# label_dis = defaultdict(int)
# for pattern in motifs:
#     for node in pattern.nodes:
#         for ins in pattern.nodes[node]['ins']:
#             G.nodes[ins]['weight'] += 1
#         label_dis[pattern.nodes[node]['label']] += 1
# # print(dis)
# print(label_dis)
# print(len(dis.keys()))
#
# motifs = match_motif(get_motif2(), G, single_edge_list, single_edge_index)
# motif_list += motifs
# # dis = defaultdict(int)
# label_dis = defaultdict(int)
# for pattern in motifs:
#     for node in pattern.nodes:
#         for ins in pattern.nodes[node]['ins']:
#             G.nodes[ins]['weight'] += 1
#         label_dis[pattern.nodes[node]['label']] += 1
# # print(dis)
# print(label_dis)
# print(len(dis.keys()))
#
# motifs = match_motif(get_motif3(), G, single_edge_list, single_edge_index)
# motif_list += motifs
# # dis = defaultdict(int)
# label_dis = defaultdict(int)
# for pattern in motifs:
#     for node in pattern.nodes:
#         for ins in pattern.nodes[node]['ins']:
#             G.nodes[ins]['weight'] += 1
#         label_dis[pattern.nodes[node]['label']] += 1
# # print(dis)
# print(label_dis)
# print(len(dis.keys()))
# print(motif_list)
# out_path = f'./{data_name}_motifs.txt'
# Utils.save_motif(motif_list, out_path)
#
# # import littleballoffur
# # i=0.2
# # sampler = littleballoffur.WeightsBasedSampler(int(G.number_of_nodes()*i))
# # newG = sampler.sample(G)
# # print(len(newG.nodes),len(newG.edges))
# Utils.save_graph_with_weights(G, "../Data/%s2weight.txt" % data_name)
#
# end_t = time.perf_counter()
# print('程序运行时间:%.2f秒' % ((end_t - st