import random
import networkx as nx
import networkit as nk
from typing import Union, List
from littleballoffur.sampler import Sampler

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class RandomEdgeSampler(Sampler):
    r"""An implementation of random edge sampling. Edges are sampled with the same
    uniform probability randomly. `"For details about the algorithm see
    this paper." <http://www.cs.ucr.edu/~michalis/PAPERS/sampling-networking-05.pdf>`_

    Args:
        number_of_edges (int): Number of edges. Default is 100.
        seed (int): Random seed. Default is 42.
    """

    def __init__(self, number_of_edges, number_of_nodes, seed: int = 1):
        self.number_of_edges = number_of_edges
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self._set_seed(seed)

    def _create_initial_edge_set(self, graph):
        """
        Choosing initial edges.
        """
        edges = self.backend.get_edges(graph)
        random.shuffle(edges)
        # edges=edges.shuffle(self.number_of_edges)
        uu=set()
        for edge in edges:
            u,v=edge
            uu.add(u)
            if len(uu) > self.number_of_nodes: break;
            uu.add(v)
            if len(uu) > self.number_of_nodes: break;
        # self._sampled_edges = random.sample(edges, self.number_of_edges)
        nodes=[]
        for u in uu:
            nodes.append(u)
        self._sampled_nodes=nodes

    def sample(self, graph: Union[NXGraph, NKGraph]) -> Union[NXGraph, NKGraph]:
        """
        Sampling edges randomly.

        Arg types:
            * **graph** *(NetworkX or NetworKit graph)* - The graph to be sampled from.

        Return types:
            * **new_graph** *(NetworkX or NetworKit graph)* - The graph of sampled edges.
        """
        self._deploy_backend(graph)
        self._check_number_of_edges(graph)
        self._create_initial_edge_set(graph)
        new_graph=self.backend.get_subgraph(graph,self._sampled_nodes)
        # new_graph = self.backend.graph_from_edgelist_new(self._sampled_edges,graph)
        return new_graph
