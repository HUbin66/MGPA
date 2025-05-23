import random
import networkx as nx
import numpy as np
import networkit as nk
from typing import Union
from littleballoffur.sampler import Sampler
from typing import List
import threading

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph

class RandomWalkBasedWeightsSampler(Sampler):

    def __init__(self, number_of_nodes: int = 100, seed: int = 42):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()

    def _create_initial_node_set(self, graph, start_nodes):
        """
        Choosing initial nodes with the highest weights.
        """
        nodes_weights = [(node, graph.nodes[node]['weight']) for node in graph.nodes()]
        sorted_nodes_weights = sorted(nodes_weights, key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes_weights[:3]]

        self._sampled_nodes = set(top_nodes)
        self._current_nodes = top_nodes

    def _do_a_step(self, graph):
        """
        Doing a single random walk step for multiple nodes.
        """
        new_current_nodes = []
        for current_node in self._current_nodes:
            neighbors = self.backend.get_neighbors(graph, current_node)
            weights = [graph.nodes[node]['weight'] for node in neighbors]
            weight_sum = sum(weights)
            weights = [weight / weight_sum for weight in weights]

            next_node = np.random.choice(neighbors, size=1, replace=False, p=weights)[0]
            new_current_nodes.append(next_node)

        for next_node in new_current_nodes:
            self._sampled_nodes.add(next_node)

        self._current_nodes = new_current_nodes

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

        threads = []
        for _ in range(3):  # Three threads
            thread = threading.Thread(target=self._do_sampling, args=(graph,))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        sampled_nodes_list = list(self._sampled_nodes)

        new_graph = self.backend.get_subgraph(graph, sampled_nodes_list)
        return new_graph

    def _do_sampling(self, graph):
        while len(self._sampled_nodes) < self.number_of_nodes:
            self._do_a_step(graph)