import sys
import pickle
from operator import gt, lt

class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.cluster_lookup = {}
        self.no_link = {}

    def add_edge(self, n1, n2, w):
        self.nodes.add(n1)
        self.nodes.add(n2)
        self.edges.setdefault(n1, {}).update({n2: w})
        self.edges.setdefault(n2, {}).update({n1: w})

    def connected_components(self, threshold=0.0, op=lt):
        nodes = set(self.nodes)
        components, visited = [], set()
        while len(nodes) > 0:
            connected, visited = self.dfs(nodes.pop(), visited, threshold, 0, op)
            connected = set(connected)
            for node in connected:
                if node in nodes:
                    nodes.remove(node)

            subgraph = Graph()
            subgraph.nodes = connected
            subgraph.no_link = self.no_link
            for s in subgraph.nodes:
                for k in self.edges.get(s, {}):
                    if k in subgraph.nodes:
                        subgraph.edges.setdefault(s, {}).update({k: self.edges[k]})
                if s in self.cluster_lookup:
                    subgraph.cluster_lookup[s] = self.cluster_lookup[s]

            components.append(subgraph)
        return components

    def dfs(self, v, visited, threshold, count, op=lt, first=None):
        aux = [v]
        visited.add(v)
        if first is None:
            first = v
        #print(self.edges.get(v, {}))
        for n in self.edges.get(v, {}):
            if n not in visited:
                #print(count)
                x, y = self.dfs(n, visited, threshold, count + 1, op, first)
                aux.extend(x)
                visited = visited.union(y)
        return aux, visited

def main(args):
    graph = Graph()
    file = open("./preprocessedData/img_sim_graph","rb")
    readGraph = pickle.load(file)
    for node1 in readGraph:
        for node2 in readGraph[node1]:
            graph.add_edge(node1, node2[0], node2[1])
    # first component
    #graph.add_edge(0, 1, 1.0)
    #graph.add_edge(1, 2, 1.0)
    #graph.add_edge(2, 0, 1.0)

    # second component
    #graph.add_edge(3, 4, 1.0)
    #graph.add_edge(4, 5, 1.0)
    #graph.add_edge(5, 3, 1.0)

    graphs = graph.connected_components(op=gt)
    print(graphs)
    print(graphs[0].nodes)
    print(graphs[1].nodes)

if __name__ == '__main__':
    main(sys.argv)