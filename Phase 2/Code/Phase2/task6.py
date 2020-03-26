import parsers as parser
import pandas as pd
import numpy as np

def task():
    
    df = None
    itemDictionary = None
    itemIndexMap = None

    #Read input <<usermoder svd 6>>  format
    inp = input("<< k >> :")
    
    try :
         k = int(inp)
    except:
        print("Error : Please enter valid value of k ")
        return

    itemDictionary = parser.parseLocationFile()
       
  
    rows, itemIndexMap = parser.processMapToDataFrame_NEW(itemDictionary,0)
    indexItemMap = res = dict((v,k) for k,v in itemIndexMap.items())
    df = pd.DataFrame(rows)
    df = df.fillna(0)

    obj_obj_mat = np.dot(df,df.T)
    U, sigma, V = np.linalg.svd(obj_obj_mat)

    ls = np.round(V, decimals = 5)

    for i , row in enumerate(ls[:k]) :
        latent_loc_tup = list(map(lambda x : ((indexItemMap[x[0]]),x[1]),enumerate(row)))
        latent_loc_tup.sort(key = lambda x : x[1],reverse = True)
        print("Latent Semantic ",1+i," :: ",latent_loc_tup)

