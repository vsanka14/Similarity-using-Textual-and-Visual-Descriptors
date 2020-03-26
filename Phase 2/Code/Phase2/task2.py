import parsers as parser
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import scipy
from collections import defaultdict
from scipy.spatial import distance
import gensim
from gensim import corpora
from sklearn.decomposition import PCA
import task2_lda

def map_id_to_location(id, location_dict):
    for term in location_dict:
        if location_dict[term]['id'] == id:
            return term

# Function to find similar users using SVD, PCA, PCA model
def task():
    
    df = None
    itemDictionary = None
    imageDictionary = defaultdict()
    userDictionary = defaultdict()
    locationDictionary = defaultdict()
    itemIndexMap = None
    indexItemMap = None
    U , sigma, V = (None, None, None)
    a,b = (300,2000)

    #Read input <<usermoder svd 6>>  format
    inp = input("Enter <<vector_space model k itemID>> :")
    inpStringArray = inp.split()
    vector_space = inpStringArray[0]
    model = inpStringArray[1]
    itemID = inpStringArray[3]
    if(vector_space.upper()=='LOCATION'):
        location_map_id = parser.parseLocationFile()
        itemID = map_id_to_location(itemID, location_map_id)
    
    try :
         k = int(inpStringArray[2])
    except:
        print("Error : Please enter valid value of k ")
        return

    # Selecting the correct vector space
    #if vector_space == "user" :
    userDictionary = parser.parseUsersFile()
        
    #elif vector_space == "image" :    
    imageDictionary = parser.parseImageFile()
        
    #elif vector_space == "location" :
    locationDictionary = parser.parseLocationFile()
       
    # Similarity calculation based on TF
    if model.upper() == 'SVD':

        rowsUser, indexUserList = parser.processMapToDataFrame(userDictionary, 0)
        df_user = pd.DataFrame(rowsUser, index = indexUserList)

        rowsImage, indexImageList = parser.processMapToDataFrame(imageDictionary, 0)
        df_image = pd.DataFrame(rowsImage, index = indexImageList)

        rowsLoc, indexLocList = parser.processMapToDataFrame(locationDictionary, 0)
        df_loc = pd.DataFrame(rowsLoc, index = indexLocList)


        df_all = pd.concat([df_user, df_image, df_loc], axis = 0)
        df_all = df_all.fillna(0)

        df_user = pd.DataFrame(df_all.iloc[:len(userDictionary),:], index = indexUserList)

        df_image = pd.DataFrame(df_all.iloc[len(userDictionary):len(userDictionary)+len(imageDictionary),:], index = indexImageList)
        
        df_loc = pd.DataFrame(df_all.iloc[len(userDictionary)+len(imageDictionary):,:], index = indexLocList)

        if vector_space == "user" :
            U_user , sigma, V = np.linalg.svd(df_user,full_matrices=False)
            V_user_df  = pd.DataFrame(V).iloc[:k,:]
            u_user_ls = pd.DataFrame(U_user, index = indexUserList).iloc[:,:k]
            U_image = pd.DataFrame(df_image)
            u_image_ls = pd.DataFrame( np.matmul(U_image.values.tolist(), V_user_df.T.values.tolist()) , index = indexImageList)
            U_loc = pd.DataFrame(df_loc, index = indexLocList)
            u_loc_ls = pd.DataFrame( np.matmul(U_loc.values.tolist(), V_user_df.T.values.tolist()) , index = indexLocList)
            id_df = pd.DataFrame(u_user_ls.loc[itemID])
            
        elif vector_space == "image" :
            U_image , sigma, V = np.linalg.svd(df_image,full_matrices=False)
            V_image_df  = pd.DataFrame(V).iloc[:k,:]
            u_image_ls = pd.DataFrame(U_image, index = indexImageList).iloc[:,:k]
            U_user = pd.DataFrame(df_user)
            u_user_ls = pd.DataFrame( np.matmul(U_user.values.tolist(), V_image_df.T.values.tolist()) , index = indexUserList)
            U_loc = pd.DataFrame(df_loc, index = indexLocList)
            u_loc_ls = pd.DataFrame( np.matmul(U_loc.values.tolist(), V_image_df.T.values.tolist()) , index = indexLocList)
            id_df = pd.DataFrame(u_image_ls.loc[itemID])

         
        elif vector_space == "location" :
            U_loc, sigma, V = np.linalg.svd(df_loc,full_matrices=False)
            V_loc_df  = pd.DataFrame(V).iloc[:k,:]
            u_loc_ls = pd.DataFrame(U_loc, index = indexLocList).iloc[:,:k]
            U_user = pd.DataFrame(df_user)
            u_user_ls = pd.DataFrame( np.matmul(U_user.values.tolist(), V_loc_df.T.values.tolist()) , index = indexUserList)
            U_image = pd.DataFrame(df_image, index = indexImageList)
            u_image_ls = pd.DataFrame( np.matmul(U_image.values.tolist(), V_loc_df.T.values.tolist()) , index = indexImageList)
            id_df = pd.DataFrame(u_loc_ls.loc[itemID])

        

        # Find User Similarity
        dd = cdist(u_user_ls, id_df.T, metric = 'euclidean')
   
        itemIdDistanceDf = pd.DataFrame(dd, index = indexUserList)
        # print(itemIdDistanceDf)
        itemIdDistanceDf = itemIdDistanceDf.sort_values(by=[0])
        print("\n\nSimilar Users")
        print(itemIdDistanceDf.head(5))
        
        dd = cdist(u_image_ls, id_df.T, metric = 'euclidean')
        # print(len(indexImageList))
        itemIdDistanceDf = pd.DataFrame(dd, index = indexImageList)
        itemIdDistanceDf = itemIdDistanceDf.sort_values(by=[0])
        print("\n\nSimilar Images")
        print(itemIdDistanceDf.head(5))
        
        dd = cdist(u_loc_ls, id_df.T, metric = 'euclidean')
        itemIdDistanceDf = pd.DataFrame(dd, index = indexLocList)
        itemIdDistanceDf = itemIdDistanceDf.sort_values(by=[0])
        print("\n\nSimilar Locations ")
        print(itemIdDistanceDf.head(5))
        
    elif model.upper() == 'LDA':
     # val4 = input("ID: ")
        vector_id = 0
        if vector_space == "user" :
            vector_id = 1
        elif vector_space == "image" :
            vector_id = 2
        elif vector_space == "location" :
            vector_id = 3
             
        
        task2_lda.task(vector_id, k, itemID)
    
    elif model.upper()=='PCA':
        rowsUser, indexUserList = parser.processMapToDataFrame(userDictionary, 0)
        df_user = pd.DataFrame(rowsUser, index = indexUserList)

        rowsImage, indexImageList = parser.processMapToDataFrame(imageDictionary, 0)
        df_image = pd.DataFrame(rowsImage, index = indexImageList)

        rowsLoc, indexLocList = parser.processMapToDataFrame(locationDictionary, 0)
        df_loc = pd.DataFrame(rowsLoc, index = indexLocList)


        df_all = pd.concat([df_user, df_image, df_loc], axis = 0)
        df_all = df_all.fillna(0)

        df_user = pd.DataFrame(df_all.iloc[:len(userDictionary),:], index = indexUserList)

        df_image = pd.DataFrame(df_all.iloc[len(userDictionary):len(userDictionary)+len(imageDictionary),:], index = indexImageList)
        
        df_loc = pd.DataFrame(df_all.iloc[len(userDictionary)+len(imageDictionary):,:], index = indexLocList)

        if vector_space == "user" :
            sklearn_pca=PCA(n_components=k)
            y_sklearn=sklearn_pca.fit(df_user)
            x_sklearn=sklearn_pca.fit_transform(df_user)
            # print(y_sklearn)
            V_user_df=pd.DataFrame(y_sklearn.components_, columns= list(df_user.columns.values))
            print(V_user_df)
            u_user_ls=pd.DataFrame(x_sklearn, index=indexUserList)
            U_image = pd.DataFrame(df_image)
            u_image_ls = pd.DataFrame( np.matmul(U_image.values.tolist(), V_user_df.T.values.tolist()) , index = indexImageList)
            U_loc = pd.DataFrame(df_loc, index = indexLocList)
            u_loc_ls = pd.DataFrame( np.matmul(U_loc.values.tolist(), V_user_df.T.values.tolist()) , index = indexLocList)
            id_df = pd.DataFrame(u_user_ls.loc[itemID])
            
        elif vector_space == "image" :
            sklearn_pca=PCA(n_components=k)
            y_sklearn=sklearn_pca.fit(df_image)
            x_sklearn=sklearn_pca.fit_transform(df_image)
            # print(y_sklearn)
            V_image_df=pd.DataFrame(y_sklearn.components_, columns= list(df_image.columns.values))
            print(V_image_df)
            u_image_ls=pd.DataFrame(x_sklearn, index=indexImageList)
            U_user = pd.DataFrame(df_user)
            u_user_ls = pd.DataFrame( np.matmul(U_user.values.tolist(), V_image_df.T.values.tolist()) , index = indexUserList)
            U_loc = pd.DataFrame(df_loc, index = indexLocList)
            u_loc_ls = pd.DataFrame( np.matmul(U_loc.values.tolist(), V_image_df.T.values.tolist()) , index = indexLocList)
            id_df = pd.DataFrame(u_image_ls.loc[itemID])

        elif vector_space == "location" :
            sklearn_pca=PCA(n_components=k)
            y_sklearn=sklearn_pca.fit(df_loc)
            x_sklearn=sklearn_pca.fit_transform(df_loc)
            # print(y_sklearn)
            V_loc_df=pd.DataFrame(y_sklearn.components_, columns= list(df_loc.columns.values))
            print(V_loc_df)
            u_loc_ls=pd.DataFrame(x_sklearn, index=indexLocList)
            U_user = pd.DataFrame(df_user)
            u_user_ls = pd.DataFrame( np.matmul(U_user.values.tolist(), V_loc_df.T.values.tolist()) , index = indexUserList)
            U_image = pd.DataFrame(df_image, index = indexImageList)
            u_image_ls = pd.DataFrame( np.matmul(U_image.values.tolist(), V_loc_df.T.values.tolist()) , index = indexImageList)
            id_df = pd.DataFrame(u_loc_ls.loc[itemID])

        # Find User Similarity
        dd = cdist(u_user_ls, id_df.T, metric = 'euclidean')
   
        itemIdDistanceDf = pd.DataFrame(dd, index = indexUserList)
        # print(itemIdDistanceDf)
        itemIdDistanceDf = itemIdDistanceDf.sort_values(by=[0])
        print("\n\nSimilar Users")
        print(itemIdDistanceDf.head(5))
        
        dd = cdist(u_image_ls, id_df.T, metric = 'euclidean')
        # print(len(indexImageList))
        itemIdDistanceDf = pd.DataFrame(dd, index = indexImageList)
        itemIdDistanceDf = itemIdDistanceDf.sort_values(by=[0])
        print("\n\nSimilar Images")
        print(itemIdDistanceDf.head(5))
        
        dd = cdist(u_loc_ls, id_df.T, metric = 'euclidean')
        itemIdDistanceDf = pd.DataFrame(dd, index = indexLocList)
        itemIdDistanceDf = itemIdDistanceDf.sort_values(by=[0])
        print("\n\nSimilar Locations ")
        print(itemIdDistanceDf.head(5))
        
        
    else:
        print('Please enter valid input.')