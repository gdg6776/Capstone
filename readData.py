import networkx as nx
import numpy as np
import pandas as pd
import pdb

class read(object):
    def __init__(self, name):
        self.name = name

    def readG(self):
        dictionary = {}

        g = nx.read_graphml(self.name)
        g = nx.Graph(g)
        g.remove_edges_from(g.selfloop_edges())

        for node in g.nodes():
            dictionary[node] = int(g.node[node]["hasrisk"].encode("utf-8"))

        attributes = ['Node', 'Risk Factor']
        data = np.array([dictionary.keys(), dictionary.values()]).transpose()

        df = pd.DataFrame(data=data, index=[i for i in range(len(dictionary.keys()))],
                          columns=attributes)

        return g, df
