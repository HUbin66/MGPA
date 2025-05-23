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



# mico
import networkx as nx
import numpy as np
import networkit as nk
from typing import Union
from littleballoffur.sampler import Sampler
from typing import List
import threading
from concurrent.futures import ThreadPoolExecutor



NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph

class EdgeWeightSampler_mico(Sampler):
    def __init__(self, number_of_nodes, seed,label_penalty,back):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed(seed)
        self.lock = threading.Lock()
        self.count=0

    def _create_initial_node_set(self, graph, start_nodes):
        """
        Choosing initial nodes with the highest weights.
        """
        # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
        # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
        nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
                         graph.nodes()]
        sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes_weights[:6]]

        self._sampled_nodes = set(top_nodes)
        self._current_nodes = top_nodes

        # nodes=random.sample(graph.nodes,3)
        # self._sampled_nodes = set(nodes)
        # self._current_nodes = nodes

    def _do_a_step(self, graph, current_node):
        neighbors = self.backend.get_neighbors(graph, current_node)
        nodes_weight=[graph.nodes[node]['node_weight'] for node in neighbors]
        weight_sum = sum(nodes_weight)
        weights = [weight / weight_sum for weight in nodes_weight]
        nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
        choose_node=neighbors[nodes_index]
        next_node=choose_node
        if next_node in self._sampled_nodes:
            self.count+=1
        self._sampled_nodes.add(next_node)
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
        # threads = []
        # for i in range(5):  # Five threads
        #     thread = threading.Thread(target=self._do_sampling, args=(graph, self._current_nodes[i]))
        #     threads.append(thread)
        #
        # for thread in threads:
        #     thread.start()
        #
        # for thread in threads:
        #     thread.join()
        # self._sampled_edges = set()
        with ThreadPoolExecutor(max_workers=6) as executor:  # 这里指定 6 个线程
            futures = []

            for i in range(6):  # Five tasks
                future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
                futures.append(future)

            for future in futures:
                future.result()

        sampled_nodes_list = list(self._sampled_nodes)
        wnode = []
        for i, node in enumerate(sampled_nodes_list):
            w = graph.nodes[node]['node_weight']
            wnode.append(w)
        dy(wnode)

        new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)

        # new_graph = nx.Graph()
        # for node in sampled_nodes_list:
        #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
        # new_graph.add_edges_from(self._sampled_edges)
        print(f"重复出现点的次数{self.count}")
        return new_graph

    def _do_sampling(self, graph, start_node):
        current_node = start_node
        while len(self._sampled_nodes) < self.number_of_nodes:
            current_node = self._do_a_step(graph, current_node)