# from queue import Queue
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# from typing import Union, List
# from collections import deque
#
#
# import threading
# # 随机+CNARW
# # import networkx as nx
# # import numpy as np
# # import networkit as nk
# # from typing import Union
# # from littleballoffur.sampler import Sampler
# # from typing import List
# # from concurrent.futures import ThreadPoolExecutor
# #
# # NKGraph = type(nk.graph.Graph())
# # NXGraph = nx.classes.graph.Graph
# #
# # class EdgeWeightSampler(Sampler):
# #     def __init__(self, number_of_nodes: int = 100, seed: int = 40):
# #         self.number_of_nodes = number_of_nodes
# #         self.seed = seed
# #         self._set_seed()
# #         self.lock = threading.Lock()
# #         self._sampler = {}
# #
# #     def _set_seed(self):
# #         import random
# #         random.seed(self.seed)
# #         np.random.seed(self.seed)
# #
# #     def _create_initial_node_set(self, graph, start_nodes):
# #         nodes_weights = [(node, graph.nodes[node]['node_weight']) for node in graph.nodes()]
# #         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
# #         top_nodes = [node for node, _ in sorted_nodes_weights[:6]]
# #         self._sampled_nodes = set(top_nodes)
# #         self._current_nodes = top_nodes
# #
# #     def _get_node_scores(self, graph, node):
# #         if node in self._sampler:  # no need to recompute
# #             return
# #         neighbors = set(self.backend.get_neighbors(graph, node))
# #         edge_weights = [graph.edges[(node, neighbor)]['edge_weight'] for neighbor in neighbors]
# #         scores = []
# #         for i, neighbor in enumerate(neighbors):
# #             fringe = set(self.backend.get_neighbors(graph, neighbor))
# #             overlap = len(neighbors.intersection(fringe))
# #             common_neighbor_score = 1.0 - (overlap) / min(
# #                 self.backend.get_degree(graph, node),
# #                 self.backend.get_degree(graph, neighbor),
# #             )
# #             edge_weight_score = edge_weights[i]
# #             combined_score = common_neighbor_score * edge_weight_score
# #             scores.append(combined_score)
# #         self._sampler[node] = {}
# #         self._sampler[node]["neighbors"] = list(neighbors)
# #         self._sampler[node]["scores"] = scores / np.sum(scores)
# #
# #     def _do_a_step(self, graph, current_node):
# #         self._get_node_scores(graph, current_node)
# #         next_node = np.random.choice(
# #             self._sampler[current_node]["neighbors"],
# #             1,
# #             replace=False,
# #             p=self._sampler[current_node]["scores"],
# #         )[0]
# #         self._sampled_nodes.add(next_node)
# #         return next_node
# #
# #     def sample(self, graph: Union[NXGraph, NKGraph], start_nodes: List[int] = None) -> Union[NXGraph, NKGraph]:
# #         self._deploy_backend(graph)
# #         self._create_initial_node_set(graph, start_nodes)
# #         with ThreadPoolExecutor(max_workers=6) as executor:  # 这里指定 6 个线程
# #             futures = []
# #             for i in range(6):  # Five tasks
# #                 future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
# #                 futures.append(future)
# #             for future in futures:
# #                 future.result()
# #         sampled_nodes_list = list(self._sampled_nodes)
# #         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
# #         return new_graph
# #
# #     def _do_sampling(self, graph, start_node):
# #         current_node = start_node
# #         while len(self._sampled_nodes) < self.number_of_nodes:
# #             current_node = self._do_a_step(graph, current_node)
#
#
# # import networkx as nx
# # import numpy as np
# # import networkit as nk
# # from typing import Union
# # from littleballoffur.sampler import Sampler
# # from typing import List
# # import threading
# # from concurrent.futures import ThreadPoolExecutor
# #
# #
# # NKGraph = type(nk.graph.Graph())
# # NXGraph = nx.classes.graph.Graph
# #
# # class EdgeWeightSampler(Sampler):
# #     def __init__(self, number_of_nodes: int = 100, seed: int = 40):
# #         self.number_of_nodes = number_of_nodes
# #         self.seed = seed
# #         self._set_seed()
# #         self.lock = threading.Lock()
# #
# #     def _create_initial_node_set(self, graph, start_nodes):
# #         """
# #         Choosing initial nodes with the highest weights.
# #         """
# #         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
# #         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
# #         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
# #                          graph.nodes()]
# #         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
# #         top_nodes = [node for node, _ in sorted_nodes_weights[:6]]
# #
# #         self._sampled_nodes = set(top_nodes)
# #         self._current_nodes = top_nodes
# #
# #         # nodes=random.sample(graph.nodes,3)
# #         # self._sampled_nodes = set(nodes)
# #         # self._current_nodes = nodes
# #
# #     def _do_a_step(self, graph, current_node):
# #         """
# #         Doing a single random walk step for a single node.
# #         """
# #         neighbors = self.backend.get_neighbors(graph, current_node)
# #         edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
# #         edges_weight=[graph.edges[edge]['edge_weight'] for edge in edges]
# #         weight_sum = sum(edges_weight)
# #         weights = [weight / weight_sum for weight in edges_weight]
# #         edge_index = np.random.choice(len(edges), size=1, replace=False, p=weights)[0]
# #         choose_edge=edges[edge_index]
# #         next_node = choose_edge[1] if current_node == choose_edge[0] else choose_edge[0]
# #
# #         # weights = [graph.nodes[node]['agg_weight'] for node in neighbors]
# #         # weight_sum = sum(weights)
# #         # weights = [weight / weight_sum for weight in weights]
# #         #
# #         # next_node = np.random.choice(neighbors, size=1, replace=False, p=weights)[0]
# #         self._sampled_nodes.add(next_node)
# #
# #         return next_node
# #
# #
# #     def sample(self, graph: Union[NXGraph, NKGraph], start_nodes: List[int] = None) -> Union[NXGraph, NKGraph]:
# #         """
# #         Sampling nodes with multiple random walks in multiple threads.
# #
# #         Arg types:
# #             * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
# #             * **start_nodes** *(List of int, optional)* - The list of start nodes.
# #
# #         Return types:
# #             * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
# #         """
# #         self._deploy_backend(graph)
# #         self._check_number_of_nodes(graph)
# #         # if start_nodes is None:
# #         #     raise ValueError("You must provide a list of start nodes.")
# #
# #         self._create_initial_node_set(graph, start_nodes)
# #
# #         # threads = []
# #         # for i in range(5):  # Five threads
# #         #     thread = threading.Thread(target=self._do_sampling, args=(graph, self._current_nodes[i]))
# #         #     threads.append(thread)
# #         #
# #         # for thread in threads:
# #         #     thread.start()
# #         #
# #         # for thread in threads:
# #         #     thread.join()
# #
# #         # self._sampled_edges = set()
# #         with ThreadPoolExecutor(max_workers=6) as executor:  # 这里指定 6 个线程
# #             futures = []
# #
# #             for i in range(6):  # Five tasks
# #                 future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
# #                 futures.append(future)
# #
# #             for future in futures:
# #                 future.result()
# #
# #         sampled_nodes_list = list(self._sampled_nodes)
# #
# #         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
# #
# #         # new_graph = nx.Graph()
# #         # for node in sampled_nodes_list:
# #         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
# #         # new_graph.add_edges_from(self._sampled_edges)
# #         return new_graph
# #
# #     def _do_sampling(self, graph, start_node):
# #         current_node = start_node
# #         while len(self._sampled_nodes) < self.number_of_nodes:
# #             current_node = self._do_a_step(graph, current_node)
#
#
#
#
#
# #
# # #我的最大权重选取
#
# # import networkx as nx
# # import numpy as np
# # import networkit as nk
# # from typing import Union
# # from littleballoffur.sampler import Sampler
# # from typing import List
# # import threading
# #
# #
# # NKGraph = type(nk.graph.Graph())
# # NXGraph = nx.classes.graph.Graph
# #
# # class EdgeWeightSampler(Sampler):
# #     def __init__(self, number_of_nodes: int = 100, seed: int = 40):
# #         self.number_of_nodes = number_of_nodes
# #         self.seed = seed
# #         self._set_seed()
# #         self.lock = threading.Lock()
# #         self._sampled_edges = set()  # 全局定义的已访问边集合
# #         self._sampled_nodes = set()  # 已访问节点集合
# #         self._current_path = []  # 当前游走路径
# #         self._hu_top = []
# #         self.js = 0  # 用于存储计数器
# #
# #
# #     def _set_seed(self):
# #         import random
# #         random.seed(self.seed)
# #         np.random.seed(self.seed)
# #
# #     def _create_initial_node_set(self, graph, start_nodes):
# #         """
# #         Choosing initial nodes with the highest weights.
# #         """
# #         nodes_weights = [(node, graph.nodes[node]['node_weight']) for node in graph.nodes()]
# #         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
# #         top_nodes = [node for node, _ in sorted_nodes_weights[:1]]
# #         self._hu_top = [node for node, _ in sorted_nodes_weights[:300000]]
# #
# #         self._sampled_nodes = set(top_nodes)
# #         self._current_path = top_nodes[:]
# #         self._current_nodes = top_nodes
# #
# #     def _do_a_step(self, graph, current_node):
# #         """
# #         Doing a single random walk step for a single node.
# #         """
# #         neighbors = self.backend.get_neighbors(graph, current_node)
# #         # 获取邻居节点的权重
# #         neighbor_weights = [graph.nodes[neighbor]['node_weight'] for neighbor in neighbors]
# #         weight_sum = sum(neighbor_weights)
# #         # 归一化权重
# #         else:
# #             weights = [weight / weight_sum for weight in neighbor_weights]
# #             # 根据节点权重随机选择下一个节点
# #             next_node = np.random.choice(neighbors, size=1, replace=False, p=weights)[0]
# #
# #         self._sampled_nodes.add(next_node)
# #         if self.debug == len(self._sampled_nodes):
# #             next_node = np.random.choice(self._hu_top)
# #             self._sampled_nodes.add(next_node)
# #             return next_node
# #         self.debug = len(self._sampled_nodes)
# #         if len(self._sampled_nodes) % 1000 == 0:
# #             print(len(self._sampled_nodes))
# #         # print(len(self._sampled_nodes))
# #         return next_node
# #
# #
# #     def _has_available_edge(self, graph, node):
# #         neighbors = self.backend.get_neighbors(graph, node)
# #         edges = [(node, neighbor) for neighbor in neighbors if graph.has_edge(node, neighbor)]
# #         edges_weight = [graph.edges[edge]['edge_weight'] for edge in edges]
# #         available_edges = [(edge, weight) for edge, weight in zip(edges, edges_weight) if edge not in self._sampled_edges]
# #         return available_edges
# #
# #     def sample(self, graph: Union[NXGraph, NKGraph], start_nodes: List[int] = None) -> Union[NXGraph, NKGraph]:
# #         """
# #         Sampling nodes with multiple random walks in a single thread.
# #         """
# #         self._deploy_backend(graph)
# #         self._check_number_of_nodes(graph)
# #         self._create_initial_node_set(graph, start_nodes)
# #
# #         for start_node in self._current_nodes:
# #             self._do_sampling(graph, start_node)
# #
# #         new_graph = nx.Graph()
# #         # new_graph.add_nodes_from(self._sampled_nodes)
# #         # for sampled_node in self._sampled_nodes:
# #         #     # 添加节点及其属性
# #         #     new_graph.add_node(sampled_node, **graph.nodes[sampled_node])
# #         # for edge in tqdm(graph.edges) :
# #         #      u, v = edge
# #         #      if u in self._sampled_nodes and v in self._sampled_nodes:
# #         #           new_graph.add_edge(u, v)
# #         new_graph = self.backend.get_subgraph(graph, self._sampled_nodes)
# #         return new_graph
# #
# #     def _do_sampling(self, graph, start_node):
# #         current_node = start_node
# #         while len(self._sampled_nodes) < self.number_of_nodes:
# #             current_node = self._do_a_step(graph, current_node)
#
# #师姐
# # import networkx as nx
# # import numpy as np
# # import networkit as nk
# # from typing import Union
# # from littleballoffur.sampler import Sampler
# # from typing import List
# # import threading
# # from concurrent.futures import ThreadPoolExecutor
# #
# #
# # NKGraph = type(nk.graph.Graph())
# # NXGraph = nx.classes.graph.Graph
# #
# # class EdgeWeightSampler(Sampler):
# #     def __init__(self, number_of_nodes: int = 100, seed: int = 40):
# #         self.number_of_nodes = number_of_nodes
# #         self.seed = seed
# #         self._set_seed()
# #         self.lock = threading.Lock()
# #
# #     def _create_initial_node_set(self, graph, start_nodes):
# #         """
# #         Choosing initial nodes with the highest weights.
# #         """
# #         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
# #         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
# #         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
# #                          graph.nodes()]
# #         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
# #         top_nodes = [node for node, _ in sorted_nodes_weights[:6]]
# #
# #         self._sampled_nodes = set(top_nodes)
# #         self._current_nodes = top_nodes
# #
# #         # nodes=random.sample(graph.nodes,3)
# #         # self._sampled_nodes = set(nodes)
# #         # self._current_nodes = nodes
# #
# #     def _do_a_step(self, graph, current_node):
# #         """
# #         Doing a single random walk step for a single node.
# #         """
# #         neighbors = self.backend.get_neighbors(graph, current_node)
# #         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
# #         nodes_weight=[graph.nodes[node]['node_weight'] for node in neighbors]
# #         weight_sum = sum(nodes_weight)
# #         weights = [weight / weight_sum for weight in nodes_weight]
# #         nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
# #         choose_node=neighbors[nodes_index]
# #         # next_node = choose_node[1] if current_node == choose_node[0] else choose_node[0]
# #         next_node=choose_node
# #         # weights = [graph.nodes[node]['agg_weight'] for node in neighbors]
# #         # weight_sum = sum(weights)
# #         # weights = [weight / weight_sum for weight in weights]
# #         #
# #         # next_node = np.random.choice(neighbors, size=1, replace=False, p=weights)[0]
# #         self._sampled_nodes.add(next_node)
# #
# #         return next_node
# #
# #
# #     def sample(self, graph: Union[NXGraph, NKGraph], start_nodes: List[int] = None) -> Union[NXGraph, NKGraph]:
# #         """
# #         Sampling nodes with multiple random walks in multiple threads.
# #
# #         Arg types:
# #             * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.
# #             * **start_nodes** *(List of int, optional)* - The list of start nodes.
# #
# #         Return types:
# #             * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
# #         """
# #         self._deploy_backend(graph)
# #         self._check_number_of_nodes(graph)
# #         # if start_nodes is None:
# #         #     raise ValueError("You must provide a list of start nodes.")
# #
# #         self._create_initial_node_set(graph, start_nodes)
# #
# #         # threads = []
# #         # for i in range(5):  # Five threads
# #         #     thread = threading.Thread(target=self._do_sampling, args=(graph, self._current_nodes[i]))
# #         #     threads.append(thread)
# #         #
# #         # for thread in threads:
# #         #     thread.start()
# #         #
# #         # for thread in threads:
# #         #     thread.join()
# #
# #         # self._sampled_edges = set()
# #         with ThreadPoolExecutor(max_workers=6) as executor:  # 这里指定 6 个线程
# #             futures = []
# #
# #             for i in range(6):  # Five tasks
# #                 future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
# #                 futures.append(future)
# #
# #             for future in futures:
# #                 future.result()
# #
# #         sampled_nodes_list = list(self._sampled_nodes)
# #
# #         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
# #
# #         # new_graph = nx.Graph()
# #         # for node in sampled_nodes_list:
# #         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
# #         # new_graph.add_edges_from(self._sampled_edges)
# #         return new_graph
# #
# #     def _do_sampling(self, graph, start_node):
# #         current_node = start_node
# #         while len(self._sampled_nodes) < self.number_of_nodes:
# #             current_node = self._do_a_step(graph, current_node)
#
# #
# from collections import deque
#
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
# from typing import List
# import threading
# from concurrent.futures import ThreadPoolExecutor
#
# import random
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
# #
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes,seed,label_penalty,back):
#         self.number_of_nodes = number_of_nodes
#         self.back=back
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#         self.top_nodes=[]
#         self.zd={}
#         self.queue = deque()
#         self.label_penalty=label_penalty
#         self.label_queue={}
#         self.queue_length=100
#         self.cnt=0
#         self.top_1000=[]
#         self._sampler = {}
#
#
#     def _get_node_scores(self, graph, node):
#         neighbors = set(self.backend.get_neighbors(graph, node))
#         scores = []
#         for i, neighbor in enumerate(neighbors):
#             fringe = set(self.backend.get_neighbors(graph, neighbor))
#             overlap = len(neighbors.intersection(fringe))
#             common_neighbor_score = 1.0 - (overlap) / min(
#                 self.backend.get_degree(graph, node),
#                 self.backend.get_degree(graph, neighbor),
#             )
#             combined_score = common_neighbor_score
#             scores.append(combined_score)
#         self._sampler[node] = {}
#         self._sampler[node]["neighbors"] = list(neighbors)
#         self._sampler[node]["scores"] = scores / np.sum(scores)
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
#         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         #避免出现某些顶点权重高但是周围点很烂的情况
#         self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
#         self.top_1000=[node for node, _ in sorted_nodes_weights[:15000]]
#         cnt = 0
#         for node in self.top_nodes:
#             if cnt > 10: break;
#             print(f"{node} , {graph.nodes[node]['label']}")
#             cnt += 1
#         self.top_nodes=random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
#         self._sampled_nodes = set(self.top_nodes)
#         self._current_nodes = self.top_nodes
#         self.zd[self.top_nodes[0]] = 1
#         self._sampled_nodes.add(self.top_nodes[0])
#         self.queue.append(self.top_nodes[0])
#         label=graph.nodes[self.top_nodes[0]]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         self._sampler = {}
#         # nodes=random.sample(graph.nodes,3)
#         # self._sampled_nodes = set(nodes)
#         # self._current_nodes = nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[]
#         for node in neighbors:
#             # if self.back==False and node in self._sampled_nodes:
#             #     nodes_weight.append(0)
#             #     continue
#             # if graph.nodes[node]['node_weight']==0 :
#             #     nodes_weight.append(0)
#             #     continue
#             label=graph.nodes[node]['label']
#             pena=self.label_penalty.get(label, 1)
#             if node not in self._sampled_nodes:
#                 nodes_weight.append(graph.nodes[node]['node_weight'])
#                 # if graph.nodes[node]['node_weight']>0.8:
#                 #     nodes_weight.append(5)
#                 # else :
#                 #     nodes_weight.append(0)
#                     # nodes_weight.append(graph.nodes[node]['node_weight'])
#             else:
#                 # 如果一个点多次被选择那么权重动态的减少
#                 graph.nodes[node]['node_weight'] = graph.nodes[node]['node_weight']
#                 nodes_weight.append(graph.nodes[node]['node_weight'])
#                 # nodes_weight.append(graph.nodes[node]['node_weight']/(3*self.zd[node])*pena)
#                 # nodes_weight.append(graph.nodes[node]['node_weight'] / (3 * self.zd[node]))
#                 # nodes_weight.append((graph.nodes[node]['node_weight'] )/ (3 * self.zd[node]))
#
#
#         weight_sum = sum(nodes_weight)
#         if  weight_sum!=0:
#             weights = [weight / weight_sum for weight in nodes_weight]
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#             choose_node = neighbors[nodes_index]
#         else:
#             self.cnt += 1
#             # 随机选择一个节点索引
#             # uniform_weights = [1.0 / len(neighbors)] * len(neighbors)
#             # nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=uniform_weights)[0]
#             # choose_node = neighbors[nodes_index]
#             # choose_node=random.choice(self.top_1000)
#             self._get_node_scores(graph, node=current_node)
#             choose_node = np.random.choice(
#                             self._sampler[current_node]["neighbors"],
#                             1,
#                             replace=False,
#                             p=self._sampler[current_node]["scores"],
#             )[0]
#         next_node = choose_node
#         self.zd[next_node] = self.zd.get(next_node, 0) + 1
#         self._sampled_nodes.add(next_node)
#         self.queue.append(next_node)
#         label=graph.nodes[next_node]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         return next_node
#
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
#         print("起始节点："+str(self.top_nodes[0]))
#         self._do_sampling(graph, self.top_nodes[0])  # Call the
#
#         sampled_nodes_list = list(self._sampled_nodes)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#         cnt=0
#         for key,val in self.zd.items():
#             if val>=3: cnt+=1
#         print("大于三次出现的个数是："+str(cnt))
#         print("weughtsum==0："+str(self.cnt))
#         # new_graph = nx.Graph()
#         # for node in sampled_nodes_list:
#         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#         # new_graph.add_edges_from(self._sampled_edges)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             if len(self.queue)%self.queue_length==0 and len(self.queue)!=0:
#                 for label,cnt in self.label_queue.items():
#                     if int(cnt)>=self.queue_length*0.3:
#                         self.label_penalty[label]=self.label_penalty.get(label,1)*0.8
#                         print(f"惩罚属性{label}")
#                 self.queue.clear()
#                 self.label_queue.clear()
#             if len(self.queue)<self.queue_length:
#                 current_node = self._do_a_step(graph, current_node)
#
#
from collections import deque

# dblp
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
# from typing import List
# import threading
# from concurrent.futures import ThreadPoolExecutor
#
# import random
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
#
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes,seed,label_penalty,back):
#         self.number_of_nodes = number_of_nodes
#         self.back=back
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#         self.top_nodes=[]
#         self.zd={}
#         self.queue = deque()
#         self.label_penalty=label_penalty
#         self.label_queue={}
#         self.queue_length=100
#         self.cnt=0
#         self.sort_node=[]
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         #避免出现某些顶点权重高但是周围点很烂的情况
#         self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
#         self.sort_node=[node for node, _ in sorted_nodes_weights]
#         cnt = 0
#         for node in self.top_nodes:
#             if cnt > 10: break;
#             print(f"{node} , {graph.nodes[node]['label']}")
#             cnt += 1
#         self.top_nodes=random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
#         self._sampled_nodes = set(self.top_nodes)
#         self._current_nodes = self.top_nodes
#         self.zd[self.top_nodes[0]] = 1
#         self._sampled_nodes.add(self.top_nodes[0])
#         self.queue.append(self.top_nodes[0])
#         label=graph.nodes[self.top_nodes[0]]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         self._sampler = {}
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[]
#         for node in neighbors:
#             if self.back==False and node in self._sampled_nodes:
#                 nodes_weight.append(0)
#                 continue
#             label=graph.nodes[node]['label']
#             # pena=self.label_penalty.get(label, 1)
#             if node not in self._sampled_nodes:
#                 if graph.nodes[node]['node_weight']<0.3:
#                     nodes_weight.append(1)
#                 else:
#                     nodes_weight.append(graph.nodes[node]['node_weight'])
#             else:
#                 # 如果一个点多次被选择那么权重动态的减少
#                 # graph.nodes[node]['node_weight'] = graph.nodes[node]['node_weight']
#                 # nodes_weight.append(graph.nodes[node]['node_weight']/(3*self.zd[node])*pena)
#                 self.cnt+=1
#                 # nodes_weight.append(graph.nodes[node]['node_weight'])
#
#         weight_sum = sum(nodes_weight)
#         if  weight_sum!=0:
#             weights = [weight / weight_sum for weight in nodes_weight]
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#
#         else:
#             uniform_weights = [1.0 / len(neighbors)] * len(neighbors)
#             self.cnt+=1
#             # 随机选择一个节点索引
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=uniform_weights)[0]
#
#         choose_node = neighbors[nodes_index]
#
#         next_node = choose_node
#         self.zd[next_node] = self.zd.get(next_node, 0) + 1
#         self._sampled_nodes.add(next_node)
#         self.queue.append(next_node)
#         label=graph.nodes[next_node]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         return next_node
#
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
#
#         self._create_initial_node_set(graph, start_nodes)
#         print("起始节点："+str(self.top_nodes[0]))
#         self._do_sampling(graph, self.top_nodes[0])  # Call the
#         sampled_nodes_list=list(self._sampled_nodes)
#         wnode = []
#         for i, node in enumerate(sampled_nodes_list):
#             w = graph.nodes[node]['node_weight']
#             wnode.append(w)
#         dy(wnode)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)


# twitch
# from collections import deque
#
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
# from typing import List
# import threading
# from concurrent.futures import ThreadPoolExecutor
#
# import random
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
# #
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes,seed,label_penalty,back=False):
#         self.number_of_nodes = number_of_nodes
#         self.back=back
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#         self.top_nodes=[]
#         self.zd={}
#         self.queue = deque()
#         self.label_penalty=label_penalty
#         self.label_queue={}
#         self.queue_length=100
#         self.cnt=0
#         self.top_1000=[]
#         self._sampler = {}
#
#
#     def _get_node_scores(self, graph, node):
#         neighbors = set(self.backend.get_neighbors(graph, node))
#         scores = []
#         for i, neighbor in enumerate(neighbors):
#             fringe = set(self.backend.get_neighbors(graph, neighbor))
#             overlap = len(neighbors.intersection(fringe))
#             common_neighbor_score = 1.0 - (overlap) / min(
#                 self.backend.get_degree(graph, node),
#                 self.backend.get_degree(graph, neighbor),
#             )
#             combined_score = common_neighbor_score
#             scores.append(combined_score)
#         self._sampler[node] = {}
#         self._sampler[node]["neighbors"] = list(neighbors)
#         self._sampler[node]["scores"] = scores / np.sum(scores)
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
#         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         #避免出现某些顶点权重高但是周围点很烂的情况
#         self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
#         cnt = 0
#         for node in self.top_nodes:
#             if cnt > 10: break;
#             print(f"{node} , {graph.nodes[node]['label']}")
#             cnt += 1
#         self.top_nodes=random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
#         self._sampled_nodes = set(self.top_nodes)
#         self._current_nodes = self.top_nodes
#         self.zd[self.top_nodes[0]] = 1
#         self._sampled_nodes.add(self.top_nodes[0])
#         self._sampler = {}
#         # nodes=random.sample(graph.nodes,3)
#         # self._sampled_nodes = set(nodes)
#         # self._current_nodes = nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[]
#         for node in neighbors:
#             # if self.back == False and node in self._sampled_nodes:
#             #     nodes_weight.append(0)
#             #     continue
#             label=graph.nodes[node]['label']
#             if graph.nodes[node]['node_weight']<0.3:
#                 nodes_weight.append(1)
#             else:
#                 nodes_weight.append(graph.nodes[node]['node_weight'])
#
#
#         weight_sum = sum(nodes_weight)
#         if  weight_sum!=0:
#             weights = [weight / weight_sum for weight in nodes_weight]
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#             choose_node = neighbors[nodes_index]
#         else:
#             self.cnt += 1
#             # 随机选择一个节点索引
#             uniform_weights = [1.0 / len(neighbors)] * len(neighbors)
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=uniform_weights)[0]
#             choose_node = neighbors[nodes_index]
#         next_node = choose_node
#         self.zd[next_node] = self.zd.get(next_node, 0) + 1
#         self._sampled_nodes.add(next_node)
#         self.queue.append(next_node)
#         label=graph.nodes[next_node]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         return next_node
#
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
#         print("起始节点："+str(self.top_nodes[0]))
#         self._do_sampling(graph, self.top_nodes[0])  # Call the
#
#         sampled_nodes_list = list(self._sampled_nodes)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#         cnt=0
#         for key,val in self.zd.items():
#             if val>=3: cnt+=1
#         print("大于三次出现的个数是："+str(cnt))
#         print("weughtsum==0："+str(self.cnt))
#         # new_graph = nx.Graph()
#         # for node in sampled_nodes_list:
#         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#         # new_graph.add_edges_from(self._sampled_edges)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)


# mico
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
#
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
#
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes, seed,label_penalty,back):
#         self.number_of_nodes = number_of_nodes
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
#         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         top_nodes = [node for node, _ in sorted_nodes_weights[:6]]
#
#         self._sampled_nodes = set(top_nodes)
#         self._current_nodes = top_nodes
#
#         # nodes=random.sample(graph.nodes,3)
#         # self._sampled_nodes = set(nodes)
#         # self._current_nodes = nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[graph.nodes[node]['node_weight'] for node in neighbors]
#         weight_sum = sum(nodes_weight)
#         weights = [weight / weight_sum for weight in nodes_weight]
#         nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#         choose_node=neighbors[nodes_index]
#         # next_node = choose_node[1] if current_node == choose_node[0] else choose_node[0]
#         next_node=choose_node
#         # weights = [graph.nodes[node]['agg_weight'] for node in neighbors]
#         # weight_sum = sum(weights)
#         # weights = [weight / weight_sum for weight in weights]
#         #
#         # next_node = np.random.choice(neighbors, size=1, replace=False, p=weights)[0]
#         self._sampled_nodes.add(next_node)
#         return next_node
#
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
#         # self._sampled_edges = set()
#         with ThreadPoolExecutor(max_workers=6) as executor:  # 这里指定 6 个线程
#             futures = []
#
#             for i in range(6):  # Five tasks
#                 future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
#                 futures.append(future)
#
#             for future in futures:
#                 future.result()
#
#         sampled_nodes_list = list(self._sampled_nodes)
#         wnode = []
#         for i, node in enumerate(sampled_nodes_list):
#             w = graph.nodes[node]['node_weight']
#             wnode.append(w)
#         dy(wnode)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#
#         # new_graph = nx.Graph()
#         # for node in sampled_nodes_list:
#         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#         # new_graph.add_edges_from(self._sampled_edges)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)


# patent
# from collections import deque
#
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
# from typing import List
# import threading
# from concurrent.futures import ThreadPoolExecutor
#
# import random
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
# #
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes,seed,label_penalty,back=False):
#         self.number_of_nodes = number_of_nodes
#         self.back=back
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#         self.top_nodes=[]
#         self.zd={}
#         self.queue = deque()
#         self.label_penalty=label_penalty
#         self.label_queue={}
#         self.queue_length=100
#         self.cnt=0
#         self.top_1000=[]
#         self._sampler = {}
#
#
#     def _get_node_scores(self, graph, node):
#         neighbors = set(self.backend.get_neighbors(graph, node))
#         scores = []
#         for i, neighbor in enumerate(neighbors):
#             fringe = set(self.backend.get_neighbors(graph, neighbor))
#             overlap = len(neighbors.intersection(fringe))
#             common_neighbor_score = 1.0 - (overlap) / min(
#                 self.backend.get_degree(graph, node),
#                 self.backend.get_degree(graph, neighbor),
#             )
#             combined_score = common_neighbor_score
#             scores.append(combined_score)
#         self._sampler[node] = {}
#         self._sampler[node]["neighbors"] = list(neighbors)
#         self._sampler[node]["scores"] = scores / np.sum(scores)
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
#         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         #避免出现某些顶点权重高但是周围点很烂的情况
#         self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
#         cnt = 0
#         for node in self.top_nodes:
#             if cnt > 10: break;
#             print(f"{node} , {graph.nodes[node]['label']}")
#             cnt += 1
#         self.top_nodes=random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
#         self._sampled_nodes = set(self.top_nodes)
#         self._current_nodes = self.top_nodes
#         self.zd[self.top_nodes[0]] = 1
#         self._sampled_nodes.add(self.top_nodes[0])
#         self._sampler = {}
#         # nodes=random.sample(graph.nodes,3)
#         # self._sampled_nodes = set(nodes)
#         # self._current_nodes = nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[]
#         for node in neighbors:
#             if self.back == False and node in self._sampled_nodes:
#                 nodes_weight.append(0)
#                 continue
#             label=graph.nodes[node]['label']
#             if graph.nodes[node]['node_weight']<0.5:
#                 nodes_weight.append(1)
#             else:
#                 nodes_weight.append(1-graph.nodes[node]['node_weight'])
#
#
#         weight_sum = sum(nodes_weight)
#         if  weight_sum!=0:
#             weights = [weight / weight_sum for weight in nodes_weight]
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#             choose_node = neighbors[nodes_index]
#         else:
#             self.cnt += 1
#             # 随机选择一个节点索引
#             self._get_node_scores(graph, node=current_node)
#             choose_node = np.random.choice(
#                     self._sampler[current_node]["neighbors"],
#                     1,
#                     replace=False,
#                     p=self._sampler[current_node]["scores"],
#             )[0]
#         next_node = choose_node
#         self.zd[next_node] = self.zd.get(next_node, 0) + 1
#         self._sampled_nodes.add(next_node)
#         self.queue.append(next_node)
#         label=graph.nodes[next_node]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         return next_node
#
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
#         print("起始节点："+str(self.top_nodes[0]))
#         self._do_sampling(graph, self.top_nodes[0])  # Call the
#
#         sampled_nodes_list = list(self._sampled_nodes)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#         cnt=0
#         for key,val in self.zd.items():
#             if val>=3: cnt+=1
#         print("大于三次出现的个数是："+str(cnt))
#         print("weughtsum==0："+str(self.cnt))
#         # new_graph = nx.Graph()
#         # for node in sampled_nodes_list:
#         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#         # new_graph.add_edges_from(self._sampled_edges)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)



# twitter
# from collections import deque
#
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
# from typing import List
# import threading
# from concurrent.futures import ThreadPoolExecutor
#
# import random
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
# #
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes,seed,back,label_penalty,yu):
#         self.number_of_nodes = number_of_nodes
#         self.back=back
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#         self.top_nodes=[]
#         self.zd={}
#         self.queue = deque()
#         self.label_penalty=label_penalty
#         self.label_queue={}
#         self.queue_length=100
#         self.cnt=0
#         self.top_1000=[]
#         self._sampler = {}
#         self._yu=yu
#
#
#     def _get_node_scores(self, graph, node):
#         neighbors = set(self.backend.get_neighbors(graph, node))
#         scores = []
#         for i, neighbor in enumerate(neighbors):
#             fringe = set(self.backend.get_neighbors(graph, neighbor))
#             overlap = len(neighbors.intersection(fringe))
#             common_neighbor_score = 1.0 - (overlap) / min(
#                 self.backend.get_degree(graph, node),
#                 self.backend.get_degree(graph, neighbor),
#             )
#             combined_score = common_neighbor_score
#             scores.append(combined_score)
#         self._sampler[node] = {}
#         self._sampler[node]["neighbors"] = list(neighbors)
#         self._sampler[node]["scores"] = scores / np.sum(scores)
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
#         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         #避免出现某些顶点权重高但是周围点很烂的情况
#         self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
#         cnt = 0
#         for node in self.top_nodes:
#             if cnt > 10: break;
#             print(f"{node} , {graph.nodes[node]['label']}")
#             cnt += 1
#         self.top_nodes=random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
#         self._sampled_nodes = set(self.top_nodes)
#         self._current_nodes = self.top_nodes
#         self.zd[self.top_nodes[0]] = 1
#         self._sampled_nodes.add(self.top_nodes[0])
#         self._sampler = {}
#         # nodes=random.sample(graph.nodes,3)
#         # self._sampled_nodes = set(nodes)
#         # self._current_nodes = nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[]
#         for node in neighbors:
#             if node in self._sampled_nodes:
#                 if self._yu:
#                     graph.nodes[node]['node_weight']=graph.nodes[node]['node_weight']/3
#                     nodes_weight.append(graph.nodes[node]['node_weight'])
#                 else:
#                     nodes_weight.append(graph.nodes[node]['node_weight']/3)
#                     # continue
#             # if graph.nodes[node]['node_weight']<0.3:
#             #     nodes_weight.append(1)
#             #     continue
#             else:
#                 nodes_weight.append(graph.nodes[node]['node_weight'])
#             # if node in self._sampled_nodes:
#             #     nodes_weight.append(graph.nodes[node]['node_weight']/(3*self.zd.get(node,1)))
#             # else:
#             #     nodes_weight.append(graph.nodes[node]['node_weight'])
#
#
#         weight_sum = sum(nodes_weight)
#         if  weight_sum!=0:
#             weights = [weight / weight_sum for weight in nodes_weight]
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#             choose_node = neighbors[nodes_index]
#         else:
#             # 随机选择一个节点索引
#             self.cnt += 1
#             uniform_weights = [1.0 / len(neighbors)] * len(neighbors)
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=uniform_weights)[0]
#             choose_node = neighbors[nodes_index]
#
#
#         next_node = choose_node
#         self.zd[next_node] = self.zd.get(next_node, 0) + 1
#         self._sampled_nodes.add(next_node)
#         self.queue.append(next_node)
#         label=graph.nodes[next_node]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         return next_node
#
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
#         print("起始节点："+str(self.top_nodes[0]))
#         self._do_sampling(graph, self.top_nodes[0])  # Call the
#
#         sampled_nodes_list = list(self._sampled_nodes)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#         cnt=0
#         for key,val in self.zd.items():
#             if val>=3: cnt+=1
#         print("大于三次出现的个数是："+str(cnt))
#         print("weughtsum==0："+str(self.cnt))
#         # new_graph = nx.Graph()
#         # for node in sampled_nodes_list:
#         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#         # new_graph.add_edges_from(self._sampled_edges)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)


# patent_new
# from collections import deque
#
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
# from typing import List
# import threading
# from concurrent.futures import ThreadPoolExecutor
#
# import random
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
# #
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes,seed,label_penalty,back=False):
#         self.number_of_nodes = number_of_nodes
#         self.back=back
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#         self.top_nodes=[]
#         self.zd={}
#         self.queue = deque()
#         self.label_penalty=label_penalty
#         self.label_queue={}
#         self.queue_length=100
#         self.cnt=0
#         self.top_1000=[]
#         self._sampler = {}
#
#
#     def _get_node_scores(self, graph, node):
#         neighbors = set(self.backend.get_neighbors(graph, node))
#         scores = []
#         for i, neighbor in enumerate(neighbors):
#             fringe = set(self.backend.get_neighbors(graph, neighbor))
#             overlap = len(neighbors.intersection(fringe))
#             common_neighbor_score = 1.0 - (overlap) / min(
#                 self.backend.get_degree(graph, node),
#                 self.backend.get_degree(graph, neighbor),
#             )
#             combined_score = common_neighbor_score
#             scores.append(combined_score)
#         self._sampler[node] = {}
#         self._sampler[node]["neighbors"] = list(neighbors)
#         self._sampler[node]["scores"] = scores / np.sum(scores)
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
#         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         #避免出现某些顶点权重高但是周围点很烂的情况
#         self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
#         cnt = 0
#         for node in self.top_nodes:
#             if cnt > 10: break;
#             print(f"{node} , {graph.nodes[node]['label']}")
#             cnt += 1
#         self.top_nodes=random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
#         self._sampled_nodes = set(self.top_nodes)
#         self._current_nodes = self.top_nodes
#         self.zd[self.top_nodes[0]] = 1
#         self._sampled_nodes.add(self.top_nodes[0])
#         self._sampler = {}
#         # nodes=random.sample(graph.nodes,3)
#         # self._sampled_nodes = set(nodes)
#         # self._current_nodes = nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[]
#         for node in neighbors:
#             if self.back == False and node in self._sampled_nodes:
#                 nodes_weight.append(0)
#                 continue
#             label=graph.nodes[node]['label']
#             if graph.nodes[node]['node_weight']<0.3:
#                 nodes_weight.append(1)
#             elif graph.nodes[node]['node_weight']>0.7:
#                 nodes_weight.append(0.3)
#             else:nodes_weight.append(graph.nodes[node]['node_weight'])
#
#
#         weight_sum = sum(nodes_weight)
#         if  weight_sum!=0:
#             weights = [weight / weight_sum for weight in nodes_weight]
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#             choose_node = neighbors[nodes_index]
#         else:
#             self.cnt += 1
#             # 随机选择一个节点索引
#             self._get_node_scores(graph, node=current_node)
#             choose_node = np.random.choice(
#                     self._sampler[current_node]["neighbors"],
#                     1,
#                     replace=False,
#                     p=self._sampler[current_node]["scores"],
#             )[0]
#         next_node = choose_node
#         self.zd[next_node] = self.zd.get(next_node, 0) + 1
#         self._sampled_nodes.add(next_node)
#         self.queue.append(next_node)
#         label=graph.nodes[next_node]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         return next_node
#
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
#         print("起始节点："+str(self.top_nodes[0]))
#         self._do_sampling(graph, self.top_nodes[0])  # Call the
#
#         sampled_nodes_list = list(self._sampled_nodes)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#         cnt=0
#         for key,val in self.zd.items():
#             if val>=3: cnt+=1
#         print("大于三次出现的个数是："+str(cnt))
#         print("weughtsum==0："+str(self.cnt))
#         # new_graph = nx.Graph()
#         # for node in sampled_nodes_list:
#         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#         # new_graph.add_edges_from(self._sampled_edges)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)


#twitter_new
# from collections import deque
# import networkx as nx
# import numpy as np
# import networkit as nk
# from typing import Union
# from littleballoffur.sampler import Sampler
# from typing import List
# import threading
# from concurrent.futures import ThreadPoolExecutor
#
# import random
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
# #
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes,seed,back,label_penalty,yu):
#         self.number_of_nodes = number_of_nodes
#         self.back=back
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#         self.top_nodes=[]
#         self.zd={}
#         self.queue = deque()
#         self.label_penalty=label_penalty
#         self.label_queue={}
#         self.queue_length=100
#         self.cnt=0
#         self.top_1000=[]
#         self._sampler = {}
#         self._yu=yu
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
#         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         #避免出现某些顶点权重高但是周围点很烂的情况
#         self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
#         cnt = 0
#         for node in self.top_nodes:
#             if cnt > 10: break;
#             print(f"{node} , {graph.nodes[node]['label']}")
#             cnt += 1
#         self.top_nodes=random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
#         self._sampled_nodes = set(self.top_nodes)
#         self._current_nodes = self.top_nodes
#         self.zd[self.top_nodes[0]] = 1
#         self._sampled_nodes.add(self.top_nodes[0])
#         self._sampler = {}
#         # nodes=random.sample(graph.nodes,3)
#         # self._sampled_nodes = set(nodes)
#         # self._current_nodes = nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[]
#         for node in neighbors:
#             if node in self._sampled_nodes:
#                 if self._yu:
#                     graph.nodes[node]['node_weight']=graph.nodes[node]['node_weight']/3
#                     nodes_weight.append(graph.nodes[node]['node_weight'])
#                 else:
#                     nodes_weight.append(graph.nodes[node]['node_weight']/3)
#                     # continue
#             # if graph.nodes[node]['node_weight']<0.3:
#             #     nodes_weight.append(1)
#             #     continue
#             else:
#                 nodes_weight.append(graph.nodes[node]['node_weight'])
#             # if node in self._sampled_nodes:
#             #     nodes_weight.append(graph.nodes[node]['node_weight']/(3*self.zd.get(node,1)))
#             # else:
#             #     nodes_weight.append(graph.nodes[node]['node_weight'])
#
#
#         weight_sum = sum(nodes_weight)
#         if  weight_sum!=0:
#             weights = [weight / weight_sum for weight in nodes_weight]
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#             choose_node = neighbors[nodes_index]
#         else:
#             # 随机选择一个节点索引
#             self.cnt += 1
#             uniform_weights = [1.0 / len(neighbors)] * len(neighbors)
#             nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=uniform_weights)[0]
#             choose_node = neighbors[nodes_index]
#
#
#         next_node = choose_node
#         self.zd[next_node] = self.zd.get(next_node, 0) + 1
#         self._sampled_nodes.add(next_node)
#         self.queue.append(next_node)
#         label=graph.nodes[next_node]['label']
#         self.label_queue[label] = self.label_queue.get(label, 0) + 1
#         return next_node
#
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
#         print("起始节点："+str(self.top_nodes[0]))
#         self._do_sampling(graph, self.top_nodes[0])  # Call the
#
#         sampled_nodes_list = list(self._sampled_nodes)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#         cnt=0
#         for key,val in self.zd.items():
#             if val>=3: cnt+=1
#         print("大于三次出现的个数是："+str(cnt))
#         print("weughtsum==0："+str(self.cnt))
#         # new_graph = nx.Graph()
#         # for node in sampled_nodes_list:
#         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#         # new_graph.add_edges_from(self._sampled_edges)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)




# mico_new
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
#
# NKGraph = type(nk.graph.Graph())
# NXGraph = nx.classes.graph.Graph
#
# class EdgeWeightSampler(Sampler):
#     def __init__(self, number_of_nodes, seed,label_penalty,back):
#         self.number_of_nodes = number_of_nodes
#         self.seed = seed
#         self._set_seed(seed)
#         self.lock = threading.Lock()
#
#     def _create_initial_node_set(self, graph, start_nodes):
#         """
#         Choosing initial nodes with the highest weights.
#         """
#         # nodes_weights = [(node, graph.nodes[node]['weight']*(1+graph.nodes[node]['agg_weight'])) for node in graph.nodes()]
#         # nodes_weights = [(node, graph.nodes[node]['weight'], graph.nodes[node]['agg_weight']) for node in graph.nodes()]
#         nodes_weights = [(node, graph.nodes[node]['node_weight'] ) for node in
#                          graph.nodes()]
#         sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
#         top_nodes = [node for node, _ in sorted_nodes_weights[:6]]
#
#         self._sampled_nodes = set(top_nodes)
#         self._current_nodes = top_nodes
#
#         # nodes=random.sample(graph.nodes,3)
#         # self._sampled_nodes = set(nodes)
#         # self._current_nodes = nodes
#
#     def _do_a_step(self, graph, current_node):
#         """
#         Doing a single random walk step for a single node.
#         """
#         neighbors = self.backend.get_neighbors(graph, current_node)
#         # nodes_weight=[]
#         # for node in neighbors:
#         #     if node in self._sampled_nodes:
#         #         nodes_weight.append(graph.nodes[node]['node_weight'])
#         #     else:
#         #         nodes_weight.append(graph.nodes[node]['node_weight'])
#
#         # edges = [(current_node, neighbor) for neighbor in neighbors if graph.has_edge(current_node, neighbor)]
#         nodes_weight=[graph.nodes[node]['node_weight'] for node in neighbors]
#         weight_sum = sum(nodes_weight)
#         weights = [weight / weight_sum for weight in nodes_weight]
#         nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
#         choose_node=neighbors[nodes_index]
#         # next_node = choose_node[1] if current_node == choose_node[0] else choose_node[0]
#         next_node=choose_node
#         # weights = [graph.nodes[node]['agg_weight'] for node in neighbors]
#         # weight_sum = sum(weights)
#         # weights = [weight / weight_sum for weight in weights]
#         #
#         # next_node = np.random.choice(neighbors, size=1, replace=False, p=weights)[0]
#         self._sampled_nodes.add(next_node)
#
#         return next_node
#
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
#         # self._sampled_edges = set()
#         with ThreadPoolExecutor(max_workers=6) as executor:  # 这里指定 6 个线程
#             futures = []
#
#             for i in range(6):  # Five tasks
#                 future = executor.submit(self._do_sampling, graph, self._current_nodes[i])
#                 futures.append(future)
#
#             for future in futures:
#                 future.result()
#
#         sampled_nodes_list = list(self._sampled_nodes)
#
#         new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
#
#         wnode=[]
#         for i,node in enumerate(sampled_nodes_list):
#             w=graph.nodes[node]['node_weight']
#             wnode.append(w)
#         dy(wnode)
#         # new_graph = nx.Graph()
#         # for node in sampled_nodes_list:
#         #     new_graph.add_node(node, label=graph.nodes[node].get('label', None))  # 添加 'label' 属性
#         # new_graph.add_edges_from(self._sampled_edges)
#         return new_graph
#
#     def _do_sampling(self, graph, start_node):
#         current_node = start_node
#         while len(self._sampled_nodes) < self.number_of_nodes:
#             current_node = self._do_a_step(graph, current_node)




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




# 统一
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

class EdgeWeightSampler(Sampler):
    def __init__(self, number_of_nodes, seed,label_penalty,back):
        # self.G=nx.Graph()
        self.back = back
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed(seed)
        self.lock = threading.Lock()
        self.top_node=-1
        self.top_nodes = []
        self.zd = {}
        self.queue = deque()
        self.label_penalty = label_penalty
        self.label_queue = {}
        self.queue_length = 100
        self.cnt = 0
        self.count=0
        self.sort_node = []
        # self.father_labe='-1'

    def _create_initial_node_set(self, graph, start_nodes):
        nodes_weights = [(node, graph.nodes[node]['node_weight']) for node in
                         graph.nodes()]
        sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
        # 避免出现某些顶点权重高但是周围点很烂的情况
        self.top_nodes = [node for node, _ in sorted_nodes_weights[:100]]
        self.sort_node = [node for node, _ in sorted_nodes_weights]
        cnt = 0
        for node in self.top_nodes:
            if cnt > 10: break;
            print(f"{node} , {graph.nodes[node]['label']}")
            cnt += 1
        self.top_node = random.sample(self.top_nodes, 1) if len(self.top_nodes) >= 1 else self.top_nodes
        self.top_node=self.top_node[0]
        self._sampled_nodes = set()
        self._current_nodes = self.top_node
        self.zd[self.top_nodes[0]] = 1
        self._sampled_nodes.add(self.top_node)
        self._sampler = {}

    def _do_a_step(self, graph, current_node):
        """
        Doing a single random walk step for a single node.
        """
        neighbors = self.backend.get_neighbors(graph, current_node)
        nodes_weight=[]
        for node in neighbors:
            node_label=graph.nodes[node]['label']
            w = graph.nodes[node]['node_weight']
            # if node_label=='2': w/=10
            if node_label == graph.nodes[current_node]['label']: w = 0
            nodes_weight.append(w)
        weight_sum = sum(nodes_weight)
        if weight_sum==0:
            self.cnt+=1
            uniform_weights = [1.0 / len(neighbors)] * len(neighbors)
            nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=uniform_weights)[0]
        else:
            weights = [weight / weight_sum for weight in nodes_weight]
            nodes_index = np.random.choice(len(neighbors), size=1, replace=False, p=weights)[0]
        choose_node=neighbors[nodes_index]
        if choose_node in self._sampled_nodes:
            self.count+=1
        next_node=choose_node
        # self.G.add_node(next_node, label=graph.nodes[next_node]['label'])
        # self.G.add_edge(current_node, next_node)
        self.zd[next_node]=self.zd.get(next_node,0) + 1
        self._sampled_nodes.add(next_node)
        # self.father_labe=graph.nodes[next_node]['label']
        return next_node


    def sample(self, graph: Union[NXGraph, NKGraph], start_nodes: List[int] = None) -> Union[NXGraph, NKGraph]:

        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        self._create_initial_node_set(graph, start_nodes)
        print("起始节点：" + str(self.top_node))
        # self.father_labe = graph.nodes[self.top_node]['label']
        self._do_sampling(graph, self.top_node)  # Call the
        sampled_nodes_list = list(self._sampled_nodes)
        new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)

        print("weightsum==0:"+str(self.cnt))
        print(f"选了{self.count}次重复的点")
        return new_graph

    def _do_sampling(self, graph, start_node):
        current_node = start_node
        while len(self._sampled_nodes) < self.number_of_nodes:
            current_node = self._do_a_step(graph, current_node)