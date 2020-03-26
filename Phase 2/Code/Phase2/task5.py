import pandas as pd
from parsers import parseLocationFile, parse_all_location_models
from scipy.linalg import svd
import numpy as np
from scipy.spatial.distance import cdist
import scipy
import task5_1 as task5_lda
from sklearn.decomposition import PCA


def perform_pca(csv_df, item_ids, csv_df_locations, k):
    sklearn_pca=PCA(n_components=k)
    x_sklearn=sklearn_pca.fit(csv_df)
    y_sklearn=sklearn_pca.fit_transform(csv_df)
    v_df=pd.DataFrame(x_sklearn.components_, columns= list(csv_df.columns.values))
    location_df=pd.DataFrame(y_sklearn, index= [item_ids, csv_df_locations])
    return location_df,v_df

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
    models_index = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
    inp = input("Enter <<location_id model k>> :")
    inpStringArray = inp.split()
    location_id = inpStringArray[0]
    model = inpStringArray[1]
    
    try :
         k = int(inpStringArray[2])
    except:
        print("Error : Please enter valid value of k ")
        return
    
    location_dict = parseLocationFile()
    input_location = map_id_to_location(location_id, location_dict)
    csv_df = parse_all_location_models(location_dict) #dataframe containing all images, locations, models and feature values
    if model.upper() == 'SVD':
        model_grouped_csv_df = csv_df.groupby([csv_df.index.get_level_values(2)]) #grouping dataframe based on model
        model_euclidean_scores = {} #creating dictionary to contain scores of all location per model
        for location in location_dict:
            model_euclidean_scores[location] = {}
        for key in model_grouped_csv_df: #iterating through grouped model dataframe 
            current_df = model_grouped_csv_df.get_group(key[0]) #current dataframe being used in iteration
            item_ids = [i[0] for i in current_df.index.tolist()] #storing item ids in list
            csv_df_locations = [i[1] for i in current_df.index.tolist()] #storing location names in list
            u_df, sigma_df, v_df = perform_svd(current_df, item_ids, csv_df_locations, k) #performing SVD transformation on current df
            print(k, ' latent semantics for ', key[0], ': \n', v_df)
            location_grouped_df = u_df.groupby([u_df.index.get_level_values(1)]) #grouping resultant U matrix based on locations
            input_df = location_grouped_df.get_group(input_location) #dataframe consisting of entries only from input location
            for inner_key in location_grouped_df: #looping through grouped location dataframe
                current_inner_df = location_grouped_df.get_group(inner_key[0]) #current dataframe inside group location dataframe loop
                column_names = [i[0] for i in current_inner_df.index.tolist()]
                input_item_ids = [i[0] for i in input_df.index.tolist()]
                dd = cdist(input_df, current_inner_df, 'euclidean') #computing pairswise similarity between items both dataframes
                euclidean_df = pd.DataFrame(dd, index=input_item_ids, columns = column_names) #dataframe containing euclidean scores between all pairs of items 
                min_euclidean_df = pd.DataFrame([euclidean_df.min(axis = 1).mean()], index=[key[0]], columns = ['score']) #only keeping minimum score in each column and averaging the columns up to get a mean score. this is the score of that locations' model
                min_euclidean_df = min_euclidean_df.rename(index=str, columns={'score': inner_key[0]}) #renaming column to location name
                model_euclidean_scores[inner_key[0]][key[0]] = (min_euclidean_df) #adding result to main dictionary which contains scores all locations per model
        master_data = [] 
        for key in model_euclidean_scores:  #iterating through dictionary to put all dataframes inside a signle dataframe
            all_data = []
            for inner_key in model_euclidean_scores[key]:
                all_data.append(model_euclidean_scores[key][inner_key].T)
            master_data.append(all_data[0].join(all_data[1:]))
        final_euclidean_scores_df = pd.concat(master_data) #dataframe containing model similarity scores between given location and all other locations
        final_euclidean_scores_df['score'] = final_euclidean_scores_df.sum(axis=1) #creating a score column to sum up values of all model scores
        # print(final_euclidean_scores_df)
        # print(k, ' latent semantics: \n', v_df)
        print('\n')
        print("Top Matching Locations: \n", final_euclidean_scores_df.sort_values(by=['score']).head(5)) #displaying top 5 results


    elif(model.upper()=='PCA'):
        model_grouped_csv_df = csv_df.groupby([csv_df.index.get_level_values(2)])
        model_euclidean_scores = {}
        for location in location_dict:
            model_euclidean_scores[location] = {}
        for xyz_index, key in enumerate(model_grouped_csv_df):
            current_df = model_grouped_csv_df.get_group(key[0])
            item_ids = [i[0] for i in current_df.index.tolist()]
            csv_df_locations = [i[1] for i in current_df.index.tolist()]
            u_df, v_df = perform_pca(current_df, item_ids, csv_df_locations, k)
            print("\n\n",models_index[xyz_index])
            print(v_df)
            location_grouped_df = u_df.groupby([u_df.index.get_level_values(1)])
            input_df = location_grouped_df.get_group(input_location)
            for inner_key in location_grouped_df:
                current_inner_df = location_grouped_df.get_group(inner_key[0])
                column_names = [i[0] for i in current_inner_df.index.tolist()]
                input_item_ids = [i[0] for i in input_df.index.tolist()]
                dd = cdist(input_df, current_inner_df, 'cosine')
                euclidean_df = pd.DataFrame(dd, index=input_item_ids, columns = column_names)
                min_euclidean_df = pd.DataFrame([euclidean_df.min(axis = 1).mean()], index=[key[0]], columns = ['score'])
                min_euclidean_df = min_euclidean_df.rename(index=str, columns={'score': inner_key[0]})
                model_euclidean_scores[inner_key[0]][key[0]] = (min_euclidean_df)
        master_data = []
        for key in model_euclidean_scores:
            all_data = []
            for inner_key in model_euclidean_scores[key]:
                all_data.append(model_euclidean_scores[key][inner_key].T)
            master_data.append(all_data[0].join(all_data[1:]))
        final_euclidean_scores_df = pd.concat(master_data)
        final_euclidean_scores_df['score'] = final_euclidean_scores_df.sum(axis=1)
        # print(final_euclidean_scores_df)
        print("Top Matching Locations: \n", final_euclidean_scores_df.sort_values(by=['score']).head(5))


    elif model.upper() == "LDA"  :
        
        task5_lda.task(location_id, k)
        
    else:
        print('Please enter valid input.')