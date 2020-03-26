import parsers as parser
import pandas as pd
from scipy.spatial.distance import cdist
models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
import pickle

def img_img_sim_pickle():
    
    df_all = None
    dfs =[]
    
    locationIdMapper = parser.location_id_mapper()
    for locationId in range(1, len(locationIdMapper)+1) :
        dfs.append(parser.parse_csv_location_all_model(locationIdMapper[locationId], models))
    
    df_all = pd.concat(dfs)
    
    #image_index_list = df_all.index.get_level_values('img_id').tolist()
    #image_location_list = df_all.index.get_level_values('location').tolist()
    #df_all.reset_index(level = 1, drop = True)    
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    image_location_dict = df_all[['location']].to_dict()
    image_location_dict = image_location_dict['location']
    image_index_list = df_all.index.get_level_values('img_id').tolist()
    del df_all['location']
    # print(df_all.shape)
    # print(len(image_location_dict))


    # print(len(image_index_list))
    
    img_dict = {img_id:num for num, img_id in enumerate(image_index_list)}
    
    
    print("Dumping img_location_mapping into pickle")
    file = open("./preprocessedData/img_loc_dict","wb")
    pickle.dump(image_location_dict, file)
    print("Dumped img_loc_dict into pickle")
    file.close()
    
    print("Dumping img_dict into pickle")
    file = open("./preprocessedData/img_dict","wb")
    pickle.dump(img_dict, file)
    print("Dumped img_dict into pickle")
    file.close()    
    
    
    print("Done creating df")
    image_image_sim_mat = cdist(df_all,df_all,'cosine')
    print("Done calculating cosine similarity")
    df_img_img = pd.DataFrame(image_image_sim_mat,index = image_index_list, columns = image_index_list)
    file = open("./preprocessedData/img_img_sim","wb")
    df_img_img.to_pickle(file)    
    print("Pickle created ")
    file.close()
    
    

    '''image_index_list = df_all.index.get_level_values('img_id').tolist()    
    print("Done creating df")
    image_image_sim_mat = cdist(df_all,df_all,'cosine')
    print("Done calculating cosine similarity")
    df_img_img = pd.DataFrame(image_image_sim_mat,index = image_index_list, columns = image_index_list)
    file = open("./preprocessedData/img_img_sim","wb")
    df_img_img.to_pickle(file)    
    print("Pickle created ")
    file.close()
    '''