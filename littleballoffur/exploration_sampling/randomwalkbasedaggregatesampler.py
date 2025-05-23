import random
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

class RandomWalkBasedAggregateSampler(Sampler):
    def __init__(self, number_of_nodes: int = 100, seed: int = 40):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()
        self.lock = threading.Lock()

    def _create_initial_node_set(self, graph, start_nodes):
        """
        Choosing initial nodes with the highest weights.
        """

        # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
        # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
        nodes_weights = [(node, graph.nodes[node]['agg_weight'] ) for node in
                         graph.nodes()]
        sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes_weights[:1]]

        self._sampled_nodes = set(top_nodes)
        self._current_nodes = top_nodes

        # nodes=random.sample(graph.nodes,3)
        # self._sampled_nodes = set(nodes)
        # self._current_nodes = nodes

    def _do_a_step(self, graph, current_node):
        """
        Doing a single random walk step for a single node.
        """
        neighbors = self.backend.get_neighbors(graph, current_node)

        # agg_weight=[graph.nodes[node]['agg_weight'] for node in neighbors]
        # agg_weight_sum=sum(agg_weight)
        # normalized_agg_weight = [weight / agg_weight_sum for weight in agg_weight]
        # weights = [graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight']) for node in neighbors]
        weights = [graph.nodes[node]['agg_weight'] for node in neighbors]
        # weights = [graph.nodes[node]['weight'] for node in neighbors]
        weight_sum = sum(weights)
        weights = [weight / weight_sum for weight in weights]

        # with self.lock:  # 使用锁来确保线程安全
        #     next_node = random.choices(neighbors, weights=weights, k=1)[0]
        #     self._sampled_nodes.add(next_node)
        next_node = np.random.choice(neighbors, size=1, replace=False, p=weights)[0]
        # next_node = np.random.choice(neighbors, size=1, replace=False)[0]
        self._sampled_nodes.add(next_node)
        # self._sampled_edges.add((current_node, next_node))

        # next_node = random.choices(neighbors, weights=weights, k=1)[0]
        # self._sampled_nodes.add(next_node)

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
        with ThreadPoolExecutor(max_workers=1) as executor:  # 这里指定 6 个线程
            futures = []

            for i in range(1):  # Five tasks
                future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
                futures.append(future)

            for future in futures:
                future.result()

        sampled_nodes_list = list(self._sampled_nodes)

        new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)

        # new_graph = nx.Graph()
        # for node in sampled_nodes_list:
        #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
        # new_graph.add_edges_from(self._sampled_edges)
        return new_graph

    def _do_sampling(self, graph, start_node):
        current_node = start_node
        while len(self._sampled_nodes) < self.number_of_nodes:
            current_node = self._do_a_step(graph, current_node)