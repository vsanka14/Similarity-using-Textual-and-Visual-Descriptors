import pandas as pd
from collections import defaultdict
import pickle
def task():
    file = open("./preprocessedData/img_img_sim","rb")
    df = pd.read_pickle(file)
    print("Read data from pickle")
    file.close()
    graph = defaultdict(list)
    #generate graph
    #Read input <<k>>  format
    inp = input("Enter <<k>> :")
    k = 0
    try :
         k = int(inp)
    except:
        print("Error : Please enter valid value of k ")
        return

    for index, row in df.iterrows():
        rowItems =list(row.iteritems())
        rowItems.sort(key = lambda x : x[1])
        graph[index] = rowItems[1:k+1]
    graph_list = list(graph.items())
    print(graph_list[:3])
    print("Dumping graph into pickle")
    file = open("./preprocessedData/img_sim_graph","wb")
    pickle.dump(graph, file)
    print("Dumped graph into pickle")
    file.close()
    #print(graph)