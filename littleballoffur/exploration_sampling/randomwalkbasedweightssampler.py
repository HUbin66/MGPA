# import random
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
#
#
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
#
# """
# slz
# 随机游走选择节点，偏向权重高的节点
# """
# class RandomWalkBasedWeightsSampler(Sampler):
#     r"""An implementation of node sampling by random walks. A simple random walker
#     which creates an induced subgraph by walking around. `"For details about the
#     algorithm see this paper." <https://ieeexplore.ieee.org/document/5462078>`_
#
#     Args:
#         number_of_nodes (int): Number of nodes. Default is 100.
#         seed (int): Random seed. Default is 42.
#     """
#
#     def __init__(self, number_of_nodes: int = 100, seed: int = 42):
#         self.number_of_nodes = number_of_nodes
#         self.seed = seed
#         self._set_seed()
#
#     def _create_initial_node_set(self, graph, start_node):
#         """
#         Choosing an initial node.
#         """
#         if start_node is not None:
#             if start_node >= 0 and start_node < self.backend.get_number_of_nodes(graph):
#                 self._current_node = start_node
#                 self._sampled_nodes = set([self._current_node])
#             else:
#                 raise ValueError("Starting node index is out of range.")
#         else:
#             self._current_node = random.choice(
#                 range(self.backend.get_number_of_nodes(graph))
#             )
#             self._sampled_nodes = set([self._current_node])
#
#     def _do_a_step(self, graph):
#         """
#         Doing a single random walk step.
#         """
#         neighbors = self.backend.get_neighbors(graph, self._current_node)
#         weights=[graph.nodes[node]['weight'] for node in neighbors]
#         # nodes = [node for node in range(self.backend.get_number_of_nodes(graph))]  # 节点编号
#         # weights = [graph.nodes[node]['weight'] for node in nodes]  # 节点权重
#         weight_sum = sum(weights)
#         weights = [weight / weight_sum for weight in weights]  # 权重归一化
#         # for node, nor s[node]['weight'] = normalized_weight
#
#         self._current_node = np.random.choice(neighbors,size=1,replace=False,p=weights)[0]
#         self._sampled_nodes.add(self._current_node)
#
#     def sample(
#         self, graph: Union[NXGraph, NKGraph], start_node: int = None
#     ) -> Union[NXGraph, NKGraph]:
#         """
#         Sampling nodes with a single random walk.
#
#         Arg types:
#             * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
#             * **start_node** *(int, optional)* - The start node.
#
#         Return types:
#             * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
#         """
#         self._deploy_backend(graph)
#         self._check_number_of_nodes(graph)
#         self._create_initial_node_set(graph, start_node)
#
#
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             self._do_a_step(graph)
#         new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
#         return new_graph



# ############################### 多线程随机游走采样
# import random
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
# from typing import List
# import threading
# from concurrent.futures import ThreadPoolExecutor
#
#
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
#
# class RandomWalkBasedWeightsSampler(Sampler):
#     def __init__(self, number_of_nodes: int = 100, seed: int = 42):
#         self.number_of_nodes = number_of_nodes
#         self.seed = seed
#         self._set_seed()
#         self.lock = threading.Lock()
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         nodes_weights = [(node, graph.nodes[node]['weight']) for node in graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         top_nodes = [node for node, _ in sorted_nodes_weights[:3]]
#
#         self._sampled_nodes = set(top_nodes)
#         self._current_nodes = top_nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         weights = [graph.nodes[node]['weight'] for node in neighbors]
#         weight_sum = sum(weights)
#         weights = [weight / weight_sum for weight in weights]
#
#         # with self.lock:  # 使用锁来确保线程安全
#         #     next_node = random.choices(neighbors, weights=weights, k=1)[0]
#         #     self._sampled_nodes.add(next_node)
#         next_node = random.choices(neighbors, weights=weights, k=1)[0]
#         self._sampled_nodes.add(next_node)
#
#         return next_node
#
#     def sample(self, graph: Union[NXGraph, NKGraph], start_nodes: List[int] = None) -> Union[NXGraph, NKGraph]:
#         """
#         Sampling nodes with multiple random walks in multiple threads.
#
#         Arg types:
#             * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
#             * **start_nodes** *(List of int, optional)* - The list of start nodes.
#
#         Return types:
#             * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
#         """
#         self._deploy_backend(graph)
#         self._check_number_of_nodes(graph)
#         # if start_nodes is None:
#         #     raise ValueError("You must provide a list of start nodes.")
#
#         self._create_initial_node_set(graph, start_nodes)
#
#         # threads = []
#         # for i in range(5):  # Five threads
#         #     thread = threading.Thread(target=self._do_sampling, args=(graph, self._current_nodes[i]))
#         #     threads.append(thread)
#         #
#         # for thread in threads:
#         #     thread.start()
#         #
#         # for thread in threads:
#         #     thread.join()
#
#         with ThreadPoolExecutor(max_workers=3) as executor:  # 这里指定 5 个线程
#             futures = []
#
#             for i in range(3):  # Five tasks
#                 future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
#                 futures.append(future)
#
#             for future in futures:
#                 future.result()
#
#         sampled_nodes_list = list(self._sampled_nodes)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)




##############################单线程  边
# import random
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
#
#
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
#
# """
# slz
# 随机游走选择节点，偏向权重高的节点
# """
# class RandomWalkBasedWeightsSampler(Sampler):
#     r"""An implementation of node sampling by random walks. A simple random walker
#     which creates an induced subgraph by walking around. `"For details about the
#     algorithm see this paper." <https://ieeexplore.ieee.org/document/5462078>`_
#
#     Args:
#         number_of_nodes (int): Number of nodes. Default is 100.
#         seed (int): Random seed. Default is 42.
#     """
#
#     def __init__(self, number_of_nodes: int = 100, seed: int = 42):
#         self.number_of_nodes = number_of_nodes
#         self.seed = seed
#         self._set_seed()
#
#     def _create_initial_node_set(self, graph, start_node):
#         """
#         Choosing an initial node.
#         """
#         if start_node is not None:
#             if start_node >= 0 and start_node < self.backend.get_number_of_nodes(graph):
#                 self._current_node = start_node
#                 self._sampled_nodes = set([self._current_node])
#             else:
#                 raise ValueError("Starting node index is out of range.")
#         else:
#             self._current_node = random.choice(
#                 range(self.backend.get_number_of_nodes(graph))
#             )
#             self._sampled_nodes = set([self._current_node])
#
#     def _do_a_step(self, graph):
#         """
#         Doing a single random walk step.
#         """
#         neighbors = self.backend.get_neighbors(graph, self._current_node)
#         weights=[graph.nodes[node]['weight'] for node in neighbors]
#         # nodes = [node for node in range(self.backend.get_number_of_nodes(graph))]  # 节点编号
#         # weights = [graph.nodes[node]['weight'] for node in nodes]  # 节点权重
#         weight_sum = sum(weights)
#         weights = [weight / weight_sum for weight in weights]  # 权重归一化
#         # for node, nor s[node]['weight'] = normalized_weight
#
#         next_node = np.random.choice(neighbors, size=1, replace=False, p=weights)[0]
#         self._sampled_nodes.add(next_node)
#         self._sampled_edges.add((self._current_node, next_node))
#         self._current_node = next_node
#
#         # self._current_node = np.random.choice(neighbors,size=1,replace=False,p=weights)[0]
#         # self._sampled_nodes.add(self._current_node)
#
#     def sample(
#         self, graph: Union[NXGraph, NKGraph], start_node: int = None
#     ) -> Union[NXGraph, NKGraph]:
#         """
#         Sampling nodes with a single random walk.
#
#         Arg types:
#             * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
#             * **start_node** *(int, optional)* - The start node.
#
#         Return types:
#             * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
#         """
#         self._deploy_backend(graph)
#         self._check_number_of_nodes(graph)
#         self._create_initial_node_set(graph, start_node)
#
#         self._sampled_edges = set()
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             self._do_a_step(graph)
#
#         # new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
#
#         new_graph = nx.Graph()
#         for node in self._sampled_nodes:
#             new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#
#         # new_graph.add_nodes_from(self._sampled_nodes)
#         new_graph.add_edges_from(self._sampled_edges)
#         return new_graph



############################### 多线程随机游走采样   边   聚合
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

class RandomWalkBasedWeightsSampler(Sampler):
    def __init__(self, number_of_nodes: int = 100, seed: int = 40):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()
        self.lock = threading.Lock()

    def _create_initial_node_set(self, graph, start_nodes):
        """
        Choosing initial nodes with the highest weights.
        """
        nodes_weights = [(node, graph.nodes[node]['weight']) for node in graph.nodes()]
        sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes_weights[:3]]

        self._sampled_nodes = set(top_nodes)
        self._current_nodes = top_nodes

    def _do_a_step(self, graph, current_node):
        """
        Doing a single random walk step for a single node.
        """
        neighbors = self.backend.get_neighbors(graph, current_node)
        weights = [graph.nodes[node]['weight'] for node in neighbors]
        weight_sum = sum(weights)
        weights = [weight / weight_sum for weight in weights]

        # with self.lock:  # 使用锁来确保线程安全
        #     next_node = random.choices(neighbors, weights=weights, k=1)[0]
        #     self._sampled_nodes.add(next_node)
        next_node = random.choices(neighbors, weights=weights, k=1)[0]
        self._sampled_nodes.add(next_node)
        self._sampled_edges.add((current_node, next_node))

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

        self._sampled_edges = set()
        with ThreadPoolExecutor(max_workers=3) as executor:  # 这里指定 3 个线程
            futures = []

            for i in range(3):  # Five tasks
                future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
                futures.append(future)

            for future in futures:
                future.result()

        sampled_nodes_list = list(self._sampled_nodes)

        # new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)

        new_graph = nx.Graph()
        for node in sampled_nodes_list:
            new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
        new_graph.add_edges_from(self._sampled_edges)
        return new_graph

    def _do_sampling(self, graph, start_node):
        current_node = start_node
        while len(self._sampled_nodes) < self.number_of_nodes:
            current_node = self._do_a_step(graph, current_node)