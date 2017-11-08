import networkx as nx
import dataf


class ego(object):
    def __init__(self, graphname, riskfactor, name, data):
        self.graphname = graphname
        self.riskfactor = riskfactor
        self.name = name
        self.data = data

    def egodata(self):
        corenumber = {}
        connectedcomponents = {}
        triangles = {}
        coefficient = {}
        egonetSize = {}

        cc = list()
        tri = list()
        coeff = list()
        egoSize = list()
        hasrisk = list()
        corenumberlist = list()


        for node in self.data:
            if self.graphname.node[node]["hasrisk"] == self.riskfactor:
                ego_graph = nx.ego_graph(self.graphname, node)
                self.computeegofeatures(corenumber, connectedcomponents, triangles, coefficient,
                                        egonetSize,cc, tri, coeff, egoSize, hasrisk,
                                        corenumberlist, ego_graph, node)
            elif self.riskfactor == -1:
                ego_graph = nx.ego_graph(self.graphname, node)
                self.computeegofeatures(corenumber, connectedcomponents, triangles, coefficient,
                                        egonetSize,cc, tri, coeff, egoSize, hasrisk,
                                        corenumberlist, ego_graph, node)

        if self.riskfactor == -1:
            frame = dataf.dataframe()
            return frame.createDataFrame(cc, tri, coeff, egoSize, hasrisk, corenumberlist, self.data)
        else:
            return connectedcomponents, triangles, coefficient, egonetSize, corenumber

    def computeegofeatures(self, corenumber, connectedcomponents, triangles, coefficient, egonetSize,
                           cc, tri, coeff, egoSize, hasrisk, corenumberlist, ego_graph, node):
        hasrisk.append(self.graphname.node[node]["hasrisk"])

        # Core number
        corenumber[node] = max(nx.core_number(ego_graph).values())
        corenumberlist.append(max(nx.core_number(ego_graph).values()))

        # egonetSize
        egonetSize[node] = ego_graph.size()
        egoSize.append(ego_graph.size())

        # Triangle count
        triangleCount = nx.triangles(ego_graph, node)
        triangles[node] = triangleCount  # adding the count for that node in dictionary
        tri.append(triangleCount)

        # Clustering co-efficients
        coeff_temp = nx.average_clustering(ego_graph)
        coefficient[node] = coeff_temp  # adding the count for that node in dictionary
        coeff.append(coeff_temp)

        # Connected components minus ego
        ego_graph.remove_node(node)
        number = nx.number_connected_components(ego_graph)  # adding the count for that node in dictionary
        connectedcomponents[node] = number
        cc.append(number)