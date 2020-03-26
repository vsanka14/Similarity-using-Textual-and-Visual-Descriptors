import pandas as pd
from parsers import parseLocationFile, parse_csv
from scipy.linalg import svd
import numpy as np
from scipy.spatial.distance import cdist
import scipy
import task3_lda
from sklearn.decomposition import PCA



def perform_pca(csv_df, item_ids, k):
    sklearn_pca=PCA(n_components=k)
    x_sklearn=sklearn_pca.fit(csv_df)
    y_sklearn=sklearn_pca.fit_transform(csv_df)
    image_df=pd.DataFrame(y_sklearn, index= list(csv_df.index.tolist()))
    V_df=pd.DataFrame(x_sklearn.components_, columns= list(csv_df.columns.values))
    return image_df, V_df

#function to perform svd which takes dataframe parsed from corresponding CSV file, image ids list and 
#k as input. after performing decomposition we only maintain k features
def perform_svd(csv_df, item_ids, k):
    u, sigma, v = np.linalg.svd(csv_df)
    u_df = (pd.DataFrame(np.round(u, decimals=5), index = item_ids))
    sigma_df = (pd.DataFrame(np.round(sigma, decimals=5)))
    v_df = (pd.DataFrame(np.round(v, decimals=5)))
    return (u_df.iloc[:, 0:k], sigma_df.iloc[0:k], v_df.iloc[0:k])

#function to calculate distance between given image and and all other images in the dataframe
#2 dfs are created -- one with just one row which is the input image id and other dataframe with all images
#distance calculated on basis of cosine similarity
def cal_sorted_distance(u_df,id, item_ids, csv_df_locations):
    id_vect = pd.DataFrame(u_df.loc[id])
    dd = cdist(u_df, id_vect.T, 'euclidean') #computing pairswise similarity between items both dataframe
    euclidean_df = pd.DataFrame(dd, index = item_ids)
    euclidean_df['location'] = csv_df_locations
    sorted_df = euclidean_df.sort_values(by=[0])
    return sorted_df

def task():    
    inp = input("Enter <<visual_descriptor_model model k imageID>> :")
    inpStringArray = inp.split()
    vd_model = inpStringArray[0]
    model = inpStringArray[1]
    imageID = int(inpStringArray[3])
    
    try :
         k = int(inpStringArray[2])
    except:
        print("Error : Please enter valid value of k ")
        return
    
    location_dict = parseLocationFile()
    csv_df = parse_csv(location_dict, vd_model) #dataframe which contains all images and feature values
    csv_df_locations = (csv_df['location'].tolist()) #storing all locations inside list
    del csv_df['location'] #deleting location row so it doesn't interfere in calculation
    # csv_df_locations = [i[1] for i in csv_df.index.tolist()]

    # Similarity calculation based on TF
    if model.upper() == 'SVD':
        item_ids = csv_df.index.tolist() #storing all image ids inside list
        u_df, sigma_df, v_df = perform_svd(csv_df, item_ids, k)
        sorted_df = cal_sorted_distance(u_df, imageID, item_ids, csv_df_locations) 
        print(k, ' latent semantics: \n', v_df)
        print('\n')
        print('Top 5 matching images: \n',sorted_df[0].head(5))
        print('\n')
        print('Top 5 matching locations: \n',sorted_df.drop_duplicates('location').head(5))
    
    elif(model.upper()=='PCA'):
        item_ids = csv_df.index.tolist()
        image_df, v_df = perform_pca(csv_df, item_ids, k)
        print(v_df)
        sorted_df = cal_sorted_distance(image_df, imageID, item_ids, csv_df_locations)
        print('\nTop 5 matching images:\n', sorted_df[0].head(5))
        print('\nTop 5 matching locations:\n', sorted_df.drop_duplicates('location').head(5))
        
        
    elif model.upper()=='LDA':
        task3_lda.task_lda(vd_model,k,imageID)

    else:
        print('Please enter valid input.')

