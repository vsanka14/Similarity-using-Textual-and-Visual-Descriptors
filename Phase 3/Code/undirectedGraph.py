from collections import defaultdict

def undirectedGraph(graph):
    result = defaultdict(defaultdict)
    for source in graph:
        destinations = graph[source]
        for dest in destinations:
            if dest[0] == source:
                continue
            if dest[0] not in result[source]:
                result[source][dest[0]] = float("inf")
            if source not in result[dest[0]]:
                result[dest[0]][source] = float("inf")
            result[source][dest[0]] = min(result[source][dest[0]], dest[1])
            result[dest[0]][source] = min(result[dest[0]][source], dest[1])

    return result
