def dy(Wnode):
    fb_hb = {}  # 初始化一个空字典来存储每个区间的计数
    v_num_hb = 0
    for i in range(len(Wnode)):
        v_num_hb += 1
        w = Wnode[i]
        # 将w四舍五入到最接近的0.1
        qz = round(w, 1)
        # 计算区间的键值
        interval_key = str(qz * 1)  # 将0.1-0.2区间转换为1，0.2-0.3转换为2，以此类推
        # 更新字典中的计数
        if interval_key in fb_hb:
            fb_hb[interval_key] += 1
        else:
            fb_hb[interval_key] = 1
    # 对字典进行排序并打印
    sorted_fb_hb = dict(sorted(fb_hb.items()))
    percentages = {key:round ((value / v_num_hb),2)   for key, value in sorted_fb_hb.items()}
    print(percentages)


# twitter
from collections import deque

import networkx as nx
import numpy as np
import networkit as nk
from typing import Union
from littleballoffur.sampler import Sampler
from typing import List
import threading
from concurrent.futures import ThreadPoolExecutor

import random
NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph
#
class EdgeWeightSampler_twitter(Sampler):
    def __init__(self, number_of_nodes,seed,back,label_penalty,yu):
        self.number_of_nodes = number_of_nodes
        self.back=back
        self.seed = seed
        self._set_seed(seed)
        self.lock = threading.Lock()
        self.top_nodes=[]
        self.zd={}
        self.queue = deque()
        self.label_penalty=label_penalty
        self.label_queue={}
        self.queue_length=100
        self.cnt=0
        self.top_1000=[]
        self._sampler = {}
        self._yu=yu
        self.count=0


    def _get_node_scores(self, graph, node):
        neighbors = set(self.backend.get_neighbors(graph, node))
        scores = []
        for i, neighbor in enumerate(neighbors):
            fringe = set(self.backend.get_neighbors(graph, neighbor))
            overlap = len(neighbors.intersection(fringe))
            common_neighbor_score = 1.0 - (overlap) / min(
                self.backend.get_degree(graph, node),
                self.backend.get_degree(graph, neighbor),
            )
            combined_score = common_neighbor_score
            scores.append(combined_score)
        self._sampler[node] = {}
        self._sampler[node]["neighbors"] = list(neighbors)
        self._sampler[node]["scores"] = scores / np.sum(scores)

    def _create_initial_node_set(self, graph, start_nodes):
        """
        Choosing initial nodes with the highest weights.
        """
        # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
        # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
        nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
                         graph.nodes()]
        sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
        #避免出现某些顶点权重高但是周围点很烂的情况
        self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
        cnt = 0
        for node in self.top_nodes:
            if cnt > 10: break;
            print(f"{node} , {graph.nodes[node]['label']}")
            cnt += 1
        self.top_nodes=random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
        self._sampled_nodes = set(self.top_nodes)
        self._current_nodes = self.top_nodes
        self.zd[self.top_nodes[0]] = 1
        self._sampled_nodes.add(self.top_nodes[0])
        self._sampler = {}
        # nodes=random.sample(graph.nodes,3)
        # self._sampled_nodes = set(nodes)
        # self._current_nodes = nodes

    def _do_a_step(self, graph, current_node):
        """
        Doing a single random walk step for a single node.
        """
        neighbors = self.backend.get_neighbors(graph, current_node)
        # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
        nodes_weight=[]
        for node in neighbors:
            if node in self._sampled_nodes:
                if self._yu:
                    graph.nodes[node]['node_weight']=graph.nodes[node]['node_weight']/3
                    nodes_weight.append(graph.nodes[node]['node_weight'])
                else:
                    nodes_weight.append(graph.nodes[node]['node_weight']/3)
                    # continue
            # if graph.nodes[node]['node_weight']<0.3:
            #     nodes_weight.append(1)
            #     continue
            else:
                nodes_weight.append(graph.nodes[node]['node_weight'])
            # if node in self._sampled_nodes:
            #     nodes_weight.append(graph.nodes[node]['node_weight']/(3*self.zd.get(node,1)))
            # else:
            #     nodes_weight.append(graph.nodes[node]['node_weight'])


        weight_sum = sum(nodes_weight)
        if  weight_sum!=0:
            weights = [weight / weight_sum for weight in nodes_weight]
            nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
            choose_node = neighbors[nodes_index]
        else:
            # 随机选择一个节点索引
            self.cnt += 1
            uniform_weights = [1.0 / len(neighbors)] * len(neighbors)
            nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=uniform_weights)[0]
            choose_node = neighbors[nodes_index]


        next_node = choose_node
        self.zd[next_node] = self.zd.get(next_node, 0) + 1
        self._sampled_nodes.add(next_node)
        self.queue.append(next_node)
        label=graph.nodes[next_node]['label']
        self.label_queue[label] = self.label_queue.get(label, 0) + 1
        if next_node in self._sampled_nodes:
            self.count+=1
        return next_node


    def sample(self, graph: Union[NXGraph, NKGraph], start_nodes: List[int] = None) -> Union[NXGraph, NKGraph]:
        """
        Sampling nodes with multiple random walks in multiple threads.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
            * **start_nodes** *(List of int, optional)* - The list of start nodes.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        # if start_nodes is None:
        #     raise ValueError("You must provide a list of start nodes.")

        self._create_initial_node_set(graph, start_nodes)

        print("起始节点："+str(self.top_nodes[0]))
        self._do_sampling(graph, self.top_nodes[0])  # Call the

        sampled_nodes_list = list(self._sampled_nodes)

        new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
        cnt=0
        for key,val in self.zd.items():
            if val>=3: cnt+=1
        print("大于三次出现的个数是："+str(cnt))
        print("重复出现点的次数："+str(self.count))
        print("weughtsum==0："+str(self.cnt))
        # new_graph = nx.Graph()
        # for node in sampled_nodes_list:
        #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
        # new_graph.add_edges_from(self._sampled_edges)
        return new_graph

    def _do_sampling(self, graph, start_node):
        current_node = start_node
        while len(self._sampled_nodes) < self.number_of_nodes:
            current_node = self._do_a_step(graph, current_node)