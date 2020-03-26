import pickle
import undirectedGraph
#import kruskalMST
import primsMST

from collections import defaultdict
from random import *
import sys


def kSpanningTree(graph, k):
    sys.setrecursionlimit(len(graph))
    graph_list = list(graph.items())
    randImg = graph_list[randint(0, len(graph) - 1)][0]
    explored = set()
    connected(graph, randImg, explored, len(graph))
    mapGraph = {}
    graphMap = {}
    i = 0
    for vertex in graph:
        mapGraph[vertex] = i
        graphMap[i] = vertex
        i += 1
    if len(explored) == len(graph):
        pMST = primsMST.Graph(len(graph))
        for source in graph:
            for dest in graph[source]:
                pMST.graph[mapGraph[source]][mapGraph[dest]] = graph[source][dest]
        minSpanTree = pMST.primMST()
        minSpanTree = sorted(minSpanTree, key = lambda x: -x[2])

        #find nodes with max edges
        '''nodes = defaultdict(int)
        node = None
        maxi = 0
        for edge in range(len(minSpanTree)):
            nodes[minSpanTree[edge][0]] += 1
            nodes[minSpanTree[edge][1]] += 1
            if nodes[minSpanTree[edge][0]] > maxi:
                maxi = nodes[minSpanTree[edge][0]]
                node = minSpanTree[edge][0]
                ind = edge
            if nodes[minSpanTree[edge][1]] > maxi:
                maxi = nodes[minSpanTree[edge][1]]
                node = minSpanTree[edge][1]
                ind = edge
        print("Max Nodes for: ", node, maxi, ind)'''

        #nodeSet = set()
        #for edge in minSpanTree:
        #    nodeSet.add(edge[0])
        #    nodeSet.add(edge[1])
        #print(len(nodeSet))

        print("Length of Min Span Tree: ", len(minSpanTree))
        result = []
        edges = []
        edgeSet = minSpanTree[1: ]
        for i in range(k - 1):
            #print("length: ", len(result))
            '''for edge in range(len(minSpanTree)):
                nodes[minSpanTree[edge][0]] += 1
                nodes[minSpanTree[edge][1]] += 1
                if nodes[minSpanTree[edge][0]] > maxi:
                    maxi = nodes[minSpanTree[edge][0]]
                    node = minSpanTree[edge][0]
                    ind = edge
                if nodes[minSpanTree[edge][1]] > maxi:
                    maxi = nodes[minSpanTree[edge][1]]
                    node = minSpanTree[edge][1]
                    ind = edge'''
            graph_list = list(graph.items())
            #randomImages.add(float("inf"))
            #randImg = float("inf")
            #while randImg in (randomImages):
                #randImg = randint(0, len(graph) - 1)
            #randomImages.add(randImg)
            randImg = randint(0, len(graph) - 1)

            head1 = minSpanTree[randImg][0]
            head2 = minSpanTree[randImg][1]
            for tree in range(len(result)):
                if head1 in result[tree]:
                    result.pop(tree)
                    edgeSet = edges.pop(tree)
                    #print("Found Tree: ", tree)
                    break
            for edge in range(len(edgeSet)):
                if edgeSet[edge] == minSpanTree[randImg]:
                    #print("EdgeSet: ", edgeSet[edge], "\tminSpanTree[i]", minSpanTree[randImg])
                    edgeSet.pop(edge)
                    break
            tree1, tree2 = create2Partitions(edgeSet, head1, head2)
            '''if len(tree1) < 50 or len(tree2) < 50:
                print(len(tree1), ", ", len(tree2))
                f.close()
                return kSpanningTree(graph, randomImages, minSpanTree)
            else:
                printF = "EdgeSet: " + str(minSpanTree[randImg]) + "\t" + str(len(tree1)) + "\t" + str(len(tree2)) + "\n" 
                f.write(printF)'''
            result.append(tree1)
            result.append(tree2)
            edges1, edges2 = create2EdgeSets(edgeSet, tree1, tree2)
            #print("Tree1: ", len(tree1))
            #print("Tree2: ", len(tree2))
            #print(len(edges1))
            #print(len(edges2))
            edges.append(edges1)
            edges.append(edges2)

        graphs = []
        for tree in range(len(result)):
            d = set()
            for node in result[tree]:
                d.add(graphMap[node])
            graphs.append(d)
        #print(len(graphs))
        totalNodes = 0
        for tree in graphs:
            totalNodes += len(tree)
            #print(len(tree))
        #print("Here totalNodes: ", totalNodes)

        totalEdges = 0
        for edge in edges:
            totalEdges += len(edge)
        #print(len(edges))
        #print("Here totalEdges: ", totalEdges)
        #print(len(graph))
        listGraphs = []
        for grph in graphs:
            currGraph = {}
            for source in grph:
                currList = list(grph.union(set(graph[source])))
                currGraph[source] = currList
            listGraphs.append(currGraph)
        
        totalNodes = 0
        for grph in listGraphs:
            totalNodes += len(grph)
            #print("Lengthy: ", len(grph))
        #print("TOTALNODES: ", totalNodes)
        #for val in listGraphs:
        #    print(val)
        #    break
        return listGraphs


def create2Partitions(tree, head1, head2):
    print("Creating Partitions")
    tree1 = set()
    tree2 = set()
    queue = [head2]
    while queue:
        currNode = queue.pop(0)
        tree1.add(currNode)
        for edge in tree:
            if edge[0] == currNode and edge[1] not in tree1:
                queue.append(edge[1])
            if edge[1] == currNode and edge[0] not in tree1:
                queue.append(edge[0])
    queue = [head1]
    while queue:
        currNode = queue.pop(0)
        tree2.add(currNode)
        for edge in tree:
            if edge[0] == currNode and edge[1] not in tree2:
                queue.append(edge[1])
            if edge[1] == currNode and edge[0] not in tree2:
                queue.append(edge[0])
    return tree1, tree2

def create2EdgeSets(edgeSet, tree1, tree2):
    print("Creating Edge Sets")
    edges1 = []
    edges2 = []
    for edges in edgeSet:
        if edges[0] in tree1 or edges[1] in tree1:
            edges1.append(edges)
        else:
            edges2.append(edges)
    return edges1, edges2
    

def connected(graph, start, explored, length):
    explored.add(start)
    for dest in graph[start]:
        if dest not in explored:
            connected(graph, dest, explored, length)
