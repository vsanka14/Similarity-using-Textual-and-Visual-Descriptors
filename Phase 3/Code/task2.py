import networkx as nx
import matplotlib.pyplot as plt
import pickle
import bfsClustering
import undirectedGraph
import kSpanningTree
from random import *

def task():
    file = open("./preprocessedData/img_sim_graph","rb")
    graph = pickle.load(file)
    print("Read data from pickle")
    file.close()
    graph_list = list(graph.items())
    print(graph_list[:3])
    try :
        algo = int(input("Input the algo no \n1. kSpanningTree\n2. bfsClustering\n"))
        k = int(input("Input the number of clusters needed: "))
    except :
        #Error out
        print("Error : Invalid Input ... Exiting")
        return
    #randImg = randint(0, len(graph_list) - 1)
    if algo == 1:
        undirGraph = undirectedGraph.undirectedGraph(graph)
        graphs = kSpanningTree.kSpanningTree(undirGraph, k)
    else:
        graphs = bfsClustering.bfs(graph, k)

    # displayGraph = pydot.Dot(graph_type='digraph', size='4.8')
    for i, grph in enumerate(graphs):
        print("Nodes in Cluster ", i, " = ", len(grph))
    color = ["red", "green", "blue", "yellow"]
    i = 0
    explored = set()
    listEdges = []
    print("Creating listEdges")
    sourceSet = set()
    for grph in graphs:
        for source in grph:
            sourceSet.add(source)
            for dest in grph[source]:
                if dest not in sourceSet:
                #print("Sorce, Dest: ", source, dest)
                    listEdges.append((source, dest))
    print("Plotting Graph")
    G = nx.DiGraph()
    G.add_edges_from(listEdges)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.show()

    '''for index, grph in enumerate(graphs):
        #displayGraph = pydot.Dot(graph_type='digraph')
        for source in grph:
            if source not in explored:
                nodeA = pydot.Node(source)
                displayGraph.add_node(nodeA)
                explored.add(source)
            for dest in grph[source]:
                if dest not in explored:
                    nodeB = pydot.Node('.', style='filled', fillcolor=color[i])
                    displayGraph.add_node(nodeB)
                    explored.add(dest)
                displayGraph.add_edge(pydot.Edge(nodeA, nodeB))
        i += 1
    displayGraph.write_png('graph.png')'''
    
    #data = displayGraph.create(prog='dot', format='png')
    #displayGraph.write('graph')
    #displayGraph.write_png('graph_' + str(index) + '.png')




    #kSpanningTree.kSpanningTree(undirGraph, set(), False)
    #print(len(graphs))

    #while True:

    #print(graph_list[:3])


