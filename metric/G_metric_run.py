import networkx as nx
import pandas as pd
import time
import os
from scipy.stats import entropy
import numpy as np
import random
from networkx.algorithms.community import greedy_modularity_communities
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon




def Evaluate(eval, G, Gs_path, G_path):
    if eval == 'CC':     # 平均局部聚类系数（AvgCC）
        # 计算所有节点的局部聚类系数
        clustering_coefficients = nx.clustering(G)

        # 计算图的平均局部聚类系数
        average_clustering = nx.average_clustering(G)
        return average_clustering
    elif eval == 'GD':   # 图密度
        density_undirected = nx.density(G)
        return density_undirected
    elif eval == 'APL':  # 平均路径长度
        if not nx.is_connected(G):
            G=max_component(G)
        avg_path_length = nx.average_shortest_path_length(G)
        return avg_path_length
    elif eval == 'ASS':  # 同配性
        r = nx.degree_assortativity_coefficient(G)
        return r
    elif eval == 'D':
        if not nx.is_connected(G):
            G=max_component(G)
        diameter = nx.diameter(G)
        return diameter
    elif eval == 'LD-SD':    #标签分布相似度  Jensen-Shannon散度
        Gs_label = {}
        G_label = {}
        with open(Gs_path) as file:
            for line in file.readlines():
                data = line.split()
                if data[0] == 'v':
                    label = str(data[2])
                    if label in Gs_label:
                        Gs_label[label] += 1
                    else:
                        Gs_label[label] = 1
        with open(G_path) as file:
            for line in file.readlines():
                data = line.split()
                if data[0] == 'v':
                    label = str(data[2])
                    if label in G_label:
                        G_label[label] += 1
                    else:
                        G_label[label] = 1

        # 确保两个分布具有相同的标签集合
        all_labels = set(Gs_label.keys()) | set(G_label.keys())
        for label in all_labels:
            Gs_label.setdefault(label, 1e-10)  # 如果标签不存在，则设置一个小概率
            G_label.setdefault(label, 1e-10)

        # 转换计数为概率
        sum_Gs = sum(Gs_label.values())
        sum_G = sum(G_label.values())
        prob_Gs = [count / sum_Gs for count in Gs_label.values()]
        prob_G = [count / sum_G for count in G_label.values()]

        # # 计算KL散度
        # kl_divergence = entropy(prob_Gs, prob_G)

        # # 创建一个二维数组（观察值表），其中每行代表一个图中的标签计数
        # obs = np.array([list(Gs_label.values()), list(G_label.values())])
        # # 执行卡方检验
        # chi2, p, dof, expected = chi2_contingency(obs)    # 卡方值 P值 自由度 期望频率度

        js_divergence = jensenshannon(prob_Gs, prob_G)
        return js_divergence

    elif eval == 'NMSE':  # 度 归一化均方偏差
        """
            计算原始图G与采样后子图G'的节点度分布之间的NMSE。
        """
        G_prime = CreateGraph(G_path)  # 原图
        theta = calculate_degree_distribution(G)
        theta_hat = calculate_degree_distribution(G_prime)

        # 确保两个分布长度相同
        length = max(len(theta), len(theta_hat))
        theta = np.resize(theta, length)
        theta_hat = np.resize(theta_hat, length)

        # 避免除以0
        theta[theta == 0] = np.nan

        # 计算NMSE
        nmse = np.nanmean(((theta_hat - theta) ** 2) / theta)
        return nmse

    elif eval == 'ACC':   #平均接近度中心性
        nodes = list(G.nodes())
        sampled_nodes = random.sample(nodes, 1000)
        total_closeness = 0
        for node in sampled_nodes:
            # 计算单个节点的接近中心性
            closeness = nx.closeness_centrality(G, u=node)
            total_closeness += closeness
        avg_closeness = total_closeness / len(sampled_nodes)
        return avg_closeness

    elif eval == 'QSC':   #社区相似度
        # 计算原图和采样图的社区直方图
        G_prime = CreateGraph(G_path)  # 原图
        hist_G = community_histogram(G)
        hist_G_prime = community_histogram(G_prime)
        # 确保两个直方图长度相同
        length = max(len(hist_G), len(hist_G_prime))
        hist_G = np.resize(hist_G, length)
        hist_G_prime = np.resize(hist_G_prime, length)
        # 计算QSC
        qsc = np.sum((hist_G - hist_G_prime) ** 2)
        return qsc

    elif eval == 'AD':
        Gs_label = {}
        G_label = {}
        with open(Gs_path) as file:
            for line in file.readlines():
                data = line.split()
                if data[0] == 'v':
                    label = str(data[2])
                    if label in Gs_label:
                        Gs_label[label] += 1
                    else:
                        Gs_label[label] = 1
        with open(G_path) as file:
            for line in file.readlines():
                data = line.split()
                if data[0] == 'v':
                    label = str(data[2])
                    if label in G_label:
                        G_label[label] += 1
                    else:
                        G_label[label] = 1

        # 确保两个分布具有相同的标签集合
        all_labels = set(Gs_label.keys()) | set(G_label.keys())
        for label in all_labels:
            Gs_label.setdefault(label, 1e-10)  # 如果标签不存在，则设置一个小概率
            G_label.setdefault(label, 1e-10)

        # 转换计数为概率
        sum_Gs = sum(Gs_label.values())
        sum_G = sum(G_label.values())
        # prob_Gs = [count / sum_Gs for count in Gs_label.values()]
        # prob_G = [count / sum_G for count in G_label.values()]

        # 计算每个标签的属性偏差
        label_deviation = {}
        for label in all_labels:
            ad = abs((Gs_label[label] / sum_Gs) - (G_label[label] / sum_G))
            label_deviation[label] = ad

        return sum(label_deviation.values())

def community_histogram(G):
    # 社区检测
    communities = greedy_modularity_communities(G)
    # 计算每个社区的大小并构建直方图
    sizes = [len(c) for c in communities]
    histogram, _ = np.histogram(sizes, bins=range(1, max(sizes) + 2), density=True)
    return histogram

def calculate_degree_distribution(G):
    """计算图G的节点度分布."""
    degrees = dict(G.degree()).values()
    max_degree = max(degrees)
    distribution = np.zeros(max_degree + 1)
    for degree in degrees:
        distribution[degree] += 1
    distribution /= sum(distribution)
    return distribution


def CreateGraph(path):
    G = nx.Graph()
    with open(path) as file:
        for line in file.readlines():
            data = line.split()
            if data[0] == 'v':
                label = str(data[2])
                G.add_node(data[1], label=label)
            elif data[0] == 'e':
                G.add_edge(data[1], data[2])
    # with open(path) as file:
    #     for line in file.readlines():
    #         data = line.split()
    #         if data[0] == 'v':
    #             label = str(data[2])
    #             G.add_node(data[1], label=label)
    #         elif data[0] == 'e':
    #             G.add_edge(data[1], data[2])
    return G

def max_component(G):
    print('remove small connected components!')
    print('begin nodes:%d, edges:%d' % (G.number_of_nodes(), G.number_of_edges()))
    maxc = len(max(nx.connected_components(G), key=lambda x: len(x)))
    remove_list = []
    for c in nx.connected_components(G):
        if len(c) != maxc:
            remove_list += c
    G.remove_nodes_from(remove_list)
    print('now nodes:%d, edges:%d' % (G.number_of_nodes(), G.number_of_edges()))
    return G

data_path=datapath = '../Data/'
# mode = ['RN','CNARW','FF','SS','RE']
mode = ['RE']
# mode = ['FF','SS','EWS','RE']
# filename = ['mico','dblp','twitter']
# filename = ['mico','dblp','twitter']
filename = ['youtube','mico','dblp','twitter','patent','twitch']

ratio=['0.050','0.100','0.150','0.200']
# ratio=['0.05','0.1','0.15000000000000002','0.2']
# ratio=['1.000']

# ratio=['0.200']

evaluator=['CC','NMSE','ACC','LD-SD']
# evaluator=['CC','ACC']

# 采样图
for i in range(1, 6):
    for file in filename:
        for r in ratio:
            # name = file + '-' + r + '.lg'    #采样图
            # name = file + '_' + r + '.lg'  # 采样图
            # G = CreateGraph(data_path +'SamplingData/'+ m + '/' + file + '/' + name)
            # file_path = '../Data/SamplingData/' + m + '/' + str(i) + '/' + file + '/500/' + name
            file_path = '../Data/' + file+'.txt'
            # file_path = '../Data/SamplingData/' + m + '/' + str(i) + '/' + file + '/' + name
            G = CreateGraph(file_path)
            # G = CreateGraph("E:\EWS采样\实验\惩罚系数beta\\beta=0.8\\"  + file + '\\' + name)
            for eval in evaluator:
                # result = Evaluate(eval, G, data_path +'SamplingData/'+m + '/' + file + '/' + file + '-' + r + '.lg', '../Data/'+file+'.txt')
                result = Evaluate(eval, G, file_path,
                                  '../Data/' + file + '.txt')
                # result = Evaluate(eval, G, '../Data/SamplingData/'+m+'/'+str(i)+'/'+file+'/'+name,
                #                   '../Data/' + file + '.txt')
                # result = Evaluate(eval, G, "E:\EWS采样\实验\惩罚系数beta\\beta=0.8\\"  + file + '\\' + name,
                #                   '../Data/' + file + '.txt')
                print(f'{file}-{eval}: {result}')
                df = pd.DataFrame({'time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())],
                                   'sampling_mode': 'G', 'filename': [file], 'matric': [eval], 'result': [result]})
                # df.to_csv('../log/exp/_metric_results.csv', index=False, mode='a', header=False)
                df.to_csv('../Data/log/exp/RE/_metric_results.csv', index=False, mode='a', header=False)


# 原图
# for file in filename:
#     name = file + '.txt'  # 原图
#     G = CreateGraph(data_path + '/' + name)
#
#     for eval in evaluator:
#         result = Evaluate(eval, G, "","")
#         print(f'{name}-{eval}: {result}')
#         df = pd.DataFrame({'time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())],
#                            'sampling_mode': ['G'], 'filename': [name], 'matric': [eval], 'result': [result]})
#         df.to_csv('../log/exp/_origin_results.csv', index=False, mode='a', header=False)