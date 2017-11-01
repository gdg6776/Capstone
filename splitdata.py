import itertools
import random
import networkx as nx
import numpy as np
from readData import read


# Total - 3268
# Not at Risk - 3033 = 1290(test) + 1743(dev)
# At risk - 235  = 100 (test) + 135(dev)

def readGraph(graph):
    mainGraph = nx.read_graphml(graph).to_undirected(reciprocal=False)
    mainGraph = nx.Graph(mainGraph)
    mainGraph.remove_edges_from(mainGraph.selfloop_edges())

    # nx.draw_networkx(mainGraph, with_labels=False, pos=nx.spring_layout(mainGraph),
    #                  edge_color='b', width=2)
    # plt.savefig("graph")
    atrisk = {}
    notatrisk = {}

    notatrisk_list = []
    atrisk_list = []

    for node in mainGraph.nodes():
        if mainGraph.node[node]["hasrisk"] == "0":
            notatrisk[node] = mainGraph.node[node]["hasrisk"]
        elif mainGraph.node[node]["hasrisk"] == "1":
            atrisk[node] = mainGraph.node[node]["hasrisk"]

    notatrisk_list = notatrisk.keys()
    atrisk_list = atrisk.keys()


    createdata(notatrisk_list, atrisk_list)


def createdata(notatrisk_list, atrisk_list):
    name = "Untitled.graphml"
    gd = read(name)
    graph1 = gd.readG()
    graph2 = gd.readG()

    notatrisk_list_1290 = random.sample(notatrisk_list,
                                        1290)  # Choosing random 1290 nodes from not at risk list (testing).
    atrisk_list_100 = random.sample(atrisk_list, 100)  # Choosing random 100 nodes from at risk list (testing).

    notatrisk_list_1743 = list(np.setdiff1d(notatrisk_list, notatrisk_list_1290,
                                            assume_unique=True))  # remaining 1743 nodes from not at risk(Dev)
    atrisk_list_135 = list(
        np.setdiff1d(atrisk_list, atrisk_list_100, assume_unique=True))  # remaining 135 nodes from at risk (Dev).

    ### Development data ###
    # Removing the 1290-Not at risk and 100-atrisk from Graph will help us build the Development data.
    for node in list(itertools.chain(notatrisk_list_1290, atrisk_list_100)):
        graph1.remove_node(node)
    nx.write_graphml(graph1, "dev.graphml")


    ### Testing Data ###
    #Removing the 1743-Not at risk and 135-atrisk from Graph will help us build the Test data.
    for node in list(itertools.chain(notatrisk_list_1743, atrisk_list_135)):
        graph2.remove_node(node)
    nx.write_graphml(graph2, "test.graphml")



def main():
    graph = "Untitled.graphml"
    readGraph(graph)


if __name__ == '__main__':
    main()