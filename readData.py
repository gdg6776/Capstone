import networkx as nx

class read(object):

    def __init__(self, name):
        self.name = name

    def readG(self):
        g = nx.read_graphml(self.name)
        g = nx.Graph(g)
        g.remove_edges_from(g.selfloop_edges())
        return g