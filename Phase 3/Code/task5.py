
import numpy as np
import pickle
from parsers import parse_all_location_models_2, parseLocationFile
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
import os

def task():
    class HashTable:
        def __init__(self, hash_size, inp_dimensions):
            self.hash_size = hash_size
            self.inp_dimensions = inp_dimensions
            self.hash_table = dict()
            self.projections = np.random.randn(self.hash_size, inp_dimensions)
        
        def __str__(self):
            return str(self.hash_table)

        def generate_hash(self, inp_vector):
            bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
            return ''.join(np.array2string(bools))

        def __setitem__(self, inp_vec, label):
            hash_value = self.generate_hash(inp_vec)
            self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]
            
        def __getitem__(self, inp_vec):
            hash_value = self.generate_hash(inp_vec)
            return self.hash_table.get(hash_value, [])

    class LSH:
        def __init__(self, num_tables, hash_size, inp_dimensions):
            self.num_tables = num_tables
            self.hash_size = hash_size
            self.inp_dimensions = inp_dimensions
            self.hash_tables = list()
            for i in range(self.num_tables):
                self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))

        def __str__(self):
            return_str = ""
            for table in self.hash_tables:
                return_str += str(pd.DataFrame.from_dict(table.hash_table, orient = 'index'))
                return_str += "\n"
            return return_str

        def __setitem__(self, inp_vec, label):
            for table in self.hash_tables:
                table[inp_vec] = label
        
        def __getitem__(self, inp_vec):
            results = list()
            for table in self.hash_tables:
                results.extend(table[inp_vec])
            return (list(set(results)), list(results))
    
    def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    inp = input("Enter <<no_of_layers no_of_hashes>> :")
    inpStringArray = inp.split()
    no_of_layers = inpStringArray[0]
    no_of_hashes = inpStringArray[1]

    images_df = parse_all_location_models_2(parseLocationFile())
    
    lsh_tables = LSH(num_tables=int(no_of_layers), hash_size=int(no_of_hashes), inp_dimensions=images_df.shape[1])

        
    for index, row in images_df.iterrows():
        lsh_tables.__setitem__(row.tolist(), index[0])
    print('Index structure created.')

    print(lsh_tables)

    img_input = input("Enter <<imageID t>>")
    inpStringArray = img_input.split()
    img_id = int(inpStringArray[0])
    t = int(inpStringArray[1])

    input_vec = images_df.loc[img_id].values.tolist()[0]
    bucket_matches, bucket_all_matches = lsh_tables.__getitem__(input_vec)
    matches_df = images_df.loc[bucket_matches]
    scores_list = []

    for index, row in matches_df.iterrows():
        cos_score = cos_sim(row.tolist(), input_vec)
        scores_list.append(cos_score)

    scores_df = pd.DataFrame(scores_list, index = matches_df.index.tolist())
    scores_df = scores_df.sort_values(by=[0], ascending=False)

    print('Top matches: \n', scores_df.head(t))
    print('Total number of unique images: \n', len(bucket_matches))
    print('Total number of overall images: \n', len(bucket_all_matches))

    try:
        os.remove('task5.html')
    except OSError:
        pass
    f = open('task4.html','a+')
    f = open('task5.html','a+')
    for index, row in scores_df.head(t).iterrows():
        filePath = "../img/"+str(index[1])+"/"+str(index[0])+".jpg"
        message = '''\
        ... <html> <body> <div> {label}  <img src={img}> </img> </div></body>.\
        ... <html>'''.format(img=filePath, label=str(index[0]))
        f.write(message)
    f.close()
        # image = Image.open(filePath)
        # image.show()
        














    