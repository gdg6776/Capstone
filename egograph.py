import networkx as nx
from dataf import dataframe


class ego(object):
    def __init__(self, graphname, riskfactor, name, filtervalue):
        self.graphname = graphname
        self.riskfactor = riskfactor
        self.name = name
        self.filtervalue = filtervalue

    def egodata(self):
        corenumber = {}
        connectedcomponents = {}
        triangles = {}
        coefficient = {}
        pathLength = {}
        atrisk = {}
        notatrisk = {}
        egonetSize = {}
        riskfactorDict = {}
        hasBridge = {}
        triangleCount = ""
        coeff = ""
        highestConnectedComponent = 0
        connectedComponentsSubGraph = ""

        notatrisk_list = []
        atrisk_list = []
        random_list = []

        nodes = list()
        cc = list()
        tri = list()
        coeff = list()
        egoSize = list()
        hasrisk = list()
        nodesList = list()
        corenumberlist = list()
        bridgelist = list()

        if not self.filtervalue:
            # nodes = filter(lambda (n, d): d[n]["hasrisk"] == self.riskfactor,
            #                self.graphname.nodes(data=True))
            for node in self.graphname.nodes():
                if self.graphname.node[node]["hasrisk"] == self.riskfactor:
                    nodes.append(node)
        else:
            nodes = self.graphname.nodes()
            # hasrisk = filter(lambda (n, d): d["hasrisk"],
            #                self.graphname.nodes(data=True))
            # for node in nodes:
            #     hasrisk.append(self.graphname.node[node]["hasrisk"])

        for node in nodes:
            ego_graph = nx.ego_graph(self.graphname, node)

            #bridges
            # print str(node) + " " +str(algorithms.bridges.has_bridges(ego_graph))

            hasrisk.append(self.graphname.node[node]["hasrisk"])

            #Core number
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

            if nx.has_bridges(ego_graph):
                hasBridge[node] = 1
                bridgelist.append(1)
            else:
                hasBridge[node] = 0
                bridgelist.append(0)

            # Average Shortest path lengths
            # copy = ego_graph.is_directed()
            # sub_graphs = nx.connected_component_subgraphs(ego_graph)  # adding the count for that node in dictionary
            # path = nx.all_pairs_shortest_path_length(ego_graph)
            # distance = sum([q.values() for q in path.values()], [])
            # pathLength[node] = float(sum(distance)) / len(distance)

        if self.filtervalue:
            frame = dataframe()
            return frame.createDataFrame(cc, tri, coeff, egoSize, hasrisk, bridgelist, corenumberlist,self.graphname.nodes())
        else:
            return connectedcomponents, triangles, coefficient, egonetSize, corenumber, hasBridge
