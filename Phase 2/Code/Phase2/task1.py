import parsers as parser
import pandas as pd
import numpy as np
from scipy.spatial import distance
import gensim
from gensim import corpora
from sklearn.decomposition import PCA

# Function to find similar users using SVD, PCA, PCA model
def task():
    
    df = None
    itemDictionary = None
    itemIndexMap = None

    #Read input <<usermoder svd 6>>  format
    inp = input("Enter <<vector_space model k>> :")
    inpStringArray = inp.split()
    vector_space = inpStringArray[0]
    model = inpStringArray[1]
    
    try :
         k = int(inpStringArray[2])
    except:
        print("Error : Please enter valid value of k ")
        return

    # Selecting the correct vector space
    if vector_space == "user" :
        itemDictionary = parser.parseUsersFile()
        
    elif vector_space == "image" :    
        itemDictionary = parser.parseImageFile()
        
    elif vector_space == "location" :
        itemDictionary = parser.parseLocationFile()
       
    # Similarity calculation based on TF
    if model.upper() == 'SVD':

        rows, itemIndexMap = parser.processMapToDataFrame(itemDictionary,0)
        df = pd.DataFrame(rows)
        df = df.fillna(0)

        U, sigma, V = np.linalg.svd(df)
    
        ls = np.round(sigma, decimals = 2)
        v_df = pd.DataFrame(V)
        
        print(v_df.iloc[:k,:])
        main_dict = {}
        for index, rows in v_df.iloc[:k, :].iterrows():
            main_dict[index] = v_df.loc[index].to_dict()
            print("\nLatent Semantics: ", index, "\n")
            sorted_dict = (sorted(main_dict[index].items(), key = lambda x:x[1], reverse = True))
            # print(sorted_dict)
            print(pd.DataFrame([sorted_dict]))
        # print(ls[:k])
    elif model.upper() == 'LDA':
        # val4 = input("ID: ")
        num_sem = k
        if(vector_space == "user"):
            a = parser.parseUsersFile()
            # print(a.keys())
            docs = list(a.keys())
            mainlist=[]
            for i in list(a.keys()):
                sublist=[]
                for k in list(a[i].keys()):
                    # print(a[i][k][0])
                    # sublist.extend([k for i in range(int(a[i][k][0]))])
                    sublist.extend([k] * int(a[i][k][0]))
                mainlist.append(sublist)
            dictionary = corpora.Dictionary(mainlist)

        elif(vector_space == "image"):
            a = parser.parseImageFile()
            docs = list(a.keys())
            # print(a.keys())
            mainlist=[]
            for i in list(a.keys()):
                sublist=[]
                for k in list(a[i].keys()):
                    # print(a[i][k][0])
                    # sublist.extend([k for i in range(int(a[i][k][0]))])
                    sublist.extend([k] * int(a[i][k][0]))
                mainlist.append(sublist)
            dictionary = corpora.Dictionary(mainlist)

        elif(vector_space == "location"):
            a = parser.parseLocationFile()
            docs = list(a.keys())

            # print(a.keys())
            mainlist=[]
            for i in list(a.keys()):
                sublist=[]
                for k in list(a[i].keys()):
                    # print(a[i][k][0])
                    # sublist.extend([k for i in range(int(a[i][k][0]))])
                    sublist.extend([k] * int(a[i][k][0]))
                mainlist.append(sublist)
            dictionary = corpora.Dictionary(mainlist)


        word2num = {k: v for v, k in enumerate(docs)}
        num2word = {v: k for v, k in enumerate(docs)}
        # print(Id_Mapping)

        # print(word2num)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in mainlist]


        Lda = gensim.models.ldamodel.LdaModel

        ldamodel = Lda(doc_term_matrix, num_topics=num_sem, id2word = dictionary, passes=1)

        print(str(ldamodel.print_topics(num_topics=num_sem)))

     
    elif model.upper() == "PCA" :
        
        rows, indexNameList = parser.processMapToDataFrame(itemDictionary,0)
        df = pd.DataFrame(rows, index = indexNameList)
        df = df.fillna(0)

        sklearn_pca=PCA(n_components=k)
        x_sklearn=sklearn_pca.fit(df)
        V_df=pd.DataFrame(x_sklearn.components_, columns= list(df.columns.values))
        print(V_df)

        main_dict = {}
        for index, rows in V_df.iterrows():
            main_dict[index] =  V_df.loc[index].to_dict()
            print("\nLatent Semantic: ",index,"\n")
            #print(sorted(main_dict[index].items(), key=lambda x: x[1], reverse=True)
            sorted_dict = sorted(main_dict[index].items(), key=lambda x: x[1], reverse=True)
            print(pd.DataFrame([sorted_dict]))

        
    else:
        print('Please enter valid input.')