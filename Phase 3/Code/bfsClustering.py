from random import *

def bfs(graph, k):
    explored = set()
    lenExp = len(list(graph.items())) // k
    n = lenExp
    result = []
    while k > 1:
        currGraph = {}                                          #graph to store curr cluster
        graph_list = list(graph.items())                        
        randImg = graph_list[randint(0, len(graph) - 1)][0]     #select random image from curr list of nodes
        queue = [randImg]                                       #queue to explore nodes of curr Graph
        currExp = set()
        while queue and (len(explored) < n):
            #print(len(explored), n, len(queue))
            currImg = queue.pop(0)
            explored.add(currImg)
            currExp.add(currImg)
            currGraph[currImg] = []
            for img in graph[currImg]:
                if img[0] not in explored:
                    queue.append(img[0])
                    #explored.add(img[0])
                    currExp.add(img[0])
                    currGraph[currImg].append(img[0])
            del graph[currImg]
        explored = explored.union(currExp)
        result.append(currGraph)
        n += lenExp
        k -= 1
    
    currGraph = {}
    for source in graph:
        #if source not in explored:
        currGraph[source] = []
        for dest in graph[source]:
            if dest[0] not in explored:
                currGraph[source].append(dest[0])
    
    result.append(currGraph)
    count = 0
    countAll = set()
    for grph in result:
        for source in grph:
            countAll.add(source)
            for dest in grph[source]:
                countAll.add(dest)
        count += len(grph)
        #print(grph)
        #print(len(grph))
    graphs = []
    for grph in result:
        d = set()
        for node in grph:
            d.add(node)
        graphs.append(d)
    #nodes = 0
    #for grph in graphs:
    #    print(len(grph))
    #    nodes += len(grph)
    #print("Total Nodes: ", nodes)
    #print(len(graphs))
    #for val in result:
    #    print(val)
    #    break
    return result