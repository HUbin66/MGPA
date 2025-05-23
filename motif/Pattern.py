import networkx as nx


class Pattern(nx.Graph):
    def get_mni(self):
        # print("gety:",[len(self.nodes[node]['ins']) for node in self.nodes])
        return min([len(self.nodes[node]['ins']) for node in self.nodes])

    def __lt__(self,p):
        return self.get_mni() < p.get_mni()