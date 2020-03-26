from sklearn import preprocessing
import parsers as parser
from scipy.spatial import distance
import gensim
from gensim import corpora
import string
import pandas as pd
from scipy.spatial.distance import cdist
def task(location_id , k_topic_cnt) :
    
    
    location_dict = parser.parseLocationFile()    
    models_index = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
    def map_id_to_location(id, location_dict):
        for term in location_dict:
            if location_dict[term]['id'] == id:
                return term
    
    
    a = parser.parseLocationFile()
    
    input_location = map_id_to_location(location_id, a)
    
    # mod = input('mod: ')
    # mod = "CM"
    maindf = parser.parse_all_location_models(a)
    # print(maindf)
    all_indexes = maindf.index.tolist()
    image_index = [i[0] for i in all_indexes]
    loc_index = [i[1] for i in all_indexes]
    mod_index = [i[2] for i in all_indexes]
    
    index_list = maindf.index.tolist()
    x = maindf.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    maindf = pd.DataFrame(x_scaled, index=[image_index, loc_index, mod_index])
    
    maindf=maindf*100
    # print("\n\n")
    # print(maindf.min())
    # print("\n\n\n\n\n\n\n\nNEWWWWWWWWWWWWWWWW")
    # print(maindf.values)
    
    maindf = maindf.round().astype(int)
    
    # print("\n\n\n\n\n\n\n\nNEWWWWWWWWWWWWWWWW")
    # print(str(maindf.index.tolist()))
    
    
    num_col = len(maindf.columns)
    
    # for t in range(1, num_col+1):
    # 	if(maindf[t].min()<0):	
    # 		maindf[t] += abs(maindf[t].min())
    
    
    # feat2num = {k: k for k in range(num_col)}
    
    # print(feat2num)
    
    # alpha = list(string.ascii_lowercase)
    
    
    
    model_grouped_csv_df = maindf.groupby([maindf.index.get_level_values(2)])
    model_euclidean_scores = {}
    for location in location_dict:
        model_euclidean_scores[location] = {}
    for i_xyz, key in enumerate(model_grouped_csv_df):
        current_df = model_grouped_csv_df.get_group(key[0])
        num_col = len(current_df.columns)
        
        item_ids = [i[0] for i in current_df.index.tolist()]
        csv_df_locations = [i[1] for i in current_df.index.tolist()]
        
        alpha = ["a"+str(i) for i in range(num_col)]
    
        for_dict_corpora=[]
        sub_dict_corpora = [alpha[i] for i in range(num_col)]
        for i in range(len(maindf.index)):
            for_dict_corpora.append(sub_dict_corpora)
        mainlist = []
        for index, rows in maindf.iterrows():
            sub_dict = maindf.loc[index].to_dict()
            sub_list = []
            for i in sub_dict:
                tup = (i-1, sub_dict[i])
                sub_list.append(tup)
            mainlist.append(sub_list)   
    
        # print(for_dict_corpora)
    
        dictionary = corpora.Dictionary(for_dict_corpora)
    
        # doc_term_matrix = [dictionary.doc2bow(doc) for doc in mainlist]
        doc_term_matrix = mainlist
        # print(dictionary.token2id)
        # print(doc_term_matrix)    
    
        Lda = gensim.models.ldamodel.LdaModel
    
        ldamodel = Lda(doc_term_matrix, num_topics = k_topic_cnt, id2word = dictionary, passes=1)
        #Print Model
        print(models_index[i_xyz])
        print(str(ldamodel.print_topics(num_topics=k_topic_cnt, num_words=3)))
        
        def get_doc_topic(corpus, model): 
            doc_topic = list() 
            for doc in corpus: 
                doc_topic.append(model.__getitem__(doc, eps=0)) 
            return doc_topic 
    
        k = get_doc_topic(doc_term_matrix, ldamodel)
    
        # print(k)
    
        dist_list = []
    
        for i in k:
            temp = []
            for j in i:
                temp.append(j[1])
            dist_list.append(temp)  
    
        # print(dist_list)
    
        temp_df = pd.DataFrame(dist_list, index=[image_index, loc_index, mod_index])
    
        # print(grouped_df)
        #print(location)
    
        grouped_df = temp_df.groupby([temp_df.index.get_level_values(1)])
    
        # location_grouped_df = u_df.groupby([u_df.index.get_level_values(1)])
        input_df = grouped_df.get_group(input_location)
        for inner_key in grouped_df:
            current_inner_df = grouped_df.get_group(inner_key[0])
            column_names = [i[0] for i in current_inner_df.index.tolist()]
            input_item_ids = [i[0] for i in input_df.index.tolist()]
            dd = cdist(input_df, current_inner_df, 'euclidean')
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
