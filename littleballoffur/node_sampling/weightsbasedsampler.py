import random
import numpy as np
import networkx as nx
import networkit as nk
from typing import Union, List
from littleballoffur.sampler import Sampler

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class WeightsBasedSampler(Sampler):      #  继承 Sampler
    r"""An implementation of degree based sampling. Nodes are sampled proportional
    to the degree centrality of nodes. `"For details about the algorithm see
    this paper." <https://arxiv.org/abs/cs/0103016>`_

    Args:
        number_of_nodes (int): Number of nodes. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 42):
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed()

    def _create_initial_node_set(self, graph: Union[NXGraph, NKGraph]) -> List[int]:
        """
        Choosing initial nodes.
        """
        nodes = [node for node in range(self.backend.get_number_of_nodes(graph))]         #  节点编号
        weights = [graph.nodes[node]['weight'] for node in nodes]             #节点权重
        # max_weight = max(weights)
        weight_sum = sum(weights)
        weights = [weight / weight_sum for weight in weights]                  #  权重归一化
        sampled_nodes = np.random.choice(nodes, size=self.number_of_nodes, replace=False, p=weights)          #  不重复采样  采样时考虑权重
        return sampled_nodes

    def sample(self, graph: Union[NXGraph, NKGraph]) -> Union[NXGraph, NKGraph]:
        """
        Sampling nodes proportional to the degree.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled nodes.
        """
        self._deploy_backend(graph)
        self._check_number_of_nodes(graph)
        sampled_nodes = self._create_initial_node_set(graph)          # 随机选择节点
        new_graph = self.backend.get_subgraph(graph, sampled_nodes)    #
        return new_graph
