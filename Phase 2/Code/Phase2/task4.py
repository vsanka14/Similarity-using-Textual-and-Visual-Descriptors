import pandas as pd
from parsers import parseLocationFile, parse_csv_with_location
from scipy.linalg import svd
import numpy as np
from scipy.spatial.distance import cdist
import scipy
import task4_lda
from sklearn.decomposition import PCA




def perform_pca(csv_df, item_ids, csv_df_locations, k):
    sklearn_pca=PCA(n_components=k)
    x_sklearn=sklearn_pca.fit(csv_df)
    y_sklearn=sklearn_pca.fit_transform(csv_df)
    location_df=pd.DataFrame(y_sklearn, index = [item_ids, csv_df_locations])
    v_df=pd.DataFrame(x_sklearn.components_, columns= list(csv_df.columns.values))
    return location_df, v_df


#function to map location id to location name
def map_id_to_location(id, location_dict):
    for term in location_dict:
        if location_dict[term]['id'] == id:
            return term

#function to perform svd which takes dataframe parsed from corresponding CSV file, image ids list, location list
#and k as input. after performing decomposition we only maintain k features
def perform_svd(csv_df, item_ids, csv_df_locations, k):
    u, sigma, v = np.linalg.svd(csv_df)
    u_df = (pd.DataFrame(np.round(u, decimals=5), index = [item_ids, csv_df_locations]))
    sigma_df = (pd.DataFrame(np.round(sigma, decimals=5)))
    v_df = (pd.DataFrame(np.round(v, decimals=5)))
    return (u_df.iloc[:, 0:k], sigma_df.iloc[0:k], v_df.iloc[0:k])
    
def task():
    inp = input("Enter <<location_id visual_descriptor_model model k>> :")
    inpStringArray = inp.split()
    location_id = inpStringArray[0]
    vd_model = inpStringArray[1]
    model = inpStringArray[2]
    
    try :
         k = int(inpStringArray[3])
    except:
        print("Error : Please enter valid value of k ")
        return
    
    location_dict = parseLocationFile()
    location = map_id_to_location(location_id, location_dict)
    csv_df = parse_csv_with_location(location_dict, vd_model) #dataframe containing all images, locations and feature values
    csv_df_locations = [i[1] for i in csv_df.index.tolist()] #storing all locations inside list
    
    if model.upper() == 'SVD':
        item_ids = [i[0] for i in csv_df.index.tolist()] #storing all image ids inside list
        u_df, sigma_df, v_df = perform_svd(csv_df, item_ids, csv_df_locations, k)
        grouped_u_df = u_df.groupby([u_df.index.get_level_values(1)]) #grouping dataframe based on location
        input_df = grouped_u_df.get_group(location) #input dataframe which consists only entries of given location
        all_data = []
        for key in grouped_u_df: #iterating through grouped dataframe to compute similarity between input dataframe and all other dataframes
            column_names = [i[0] for i in grouped_u_df.get_group(key[0]).index.tolist()]
            input_item_ids = [i[0] for i in grouped_u_df.get_group(location).index.tolist()]
            dd = cdist(input_df, grouped_u_df.get_group(key[0]), 'euclidean') #computing pairswise similarity between items both dataframes
            euclidean_df = pd.DataFrame(dd, index=input_item_ids, columns = column_names) #dataframe containing euclidean scores between all pairs of items 
            min_euclidean_df = pd.DataFrame([euclidean_df.min(axis = 1).mean()], index=[key[0]], columns = ['score']) #only keeping minimum score in each column and averaging the columns up to get a mean score. this is the score of that locations' model
            all_data.append(min_euclidean_df)
        all_euclidean_scores_df = pd.concat(all_data) #dataframe with euclidean scores of all locations
        print(k, ' latent semantics: \n', v_df)
        print('\n')
        print("Top 5 Similar Locations: \n", all_euclidean_scores_df.sort_values(by=['score']).head(5)) #display top 5 locations
        # sorted_df = cal_sorted_distance(u_df, imageID, item_ids, csv_df_locations)
        # print('Top 5 matching images: \n',sorted_df[0].head(5))
        # print('Top 5 matching locations: \n',sorted_df.drop_duplicates('location')['location'].head(5))

    elif(model.upper()=='PCA'):
        item_ids = [i[0] for i in csv_df.index.tolist()]
        location_df, v_df=perform_pca(csv_df,item_ids,csv_df_locations,k)
        print(v_df)
        grouped_location_df = location_df.groupby([location_df.index.get_level_values(1)])
        input_df = grouped_location_df.get_group(location)
        all_data=[]
        for key in grouped_location_df:
            column_names = [i[0] for i in grouped_location_df.get_group(key[0]).index.tolist()]
            input_item_ids = [i[0] for i in grouped_location_df.get_group(location).index.tolist()]
            dd = cdist(input_df, grouped_location_df.get_group(key[0]),'cosine')
            cosine_df = pd.DataFrame(dd, index = input_item_ids, columns = column_names)
            min_cosine_df = pd.DataFrame([cosine_df.min(axis=1).mean()], index=[key[0]], columns=['score'])
            all_data.append(min_cosine_df)
        all_euclidean_scores_df=pd.concat(all_data)
        print("Top 5 Similar Locations: \n", all_euclidean_scores_df.sort_values(by=['score']).head(5))
        
        
    elif model.upper() == 'LDA':
        task4_lda.task_lda(vd_model,k,location_id)	
    else:
        print('Please enter valid input.')