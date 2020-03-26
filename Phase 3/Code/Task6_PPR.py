# import some stuff
import numpy as np
import pandas as pd
import pickle

def task():
        file = open("../preprocessedData/img_sim_graph","rb")
        graph = pickle.load(file)
        # print("Read data from pickle")
        file.close()

        file = open("../preprocessedData/img_dict","rb")
        img_dict = pickle.load(file)
        # print("Read img_dict from pickle")
        file.close()

        graph_list = graph.items()

        inputFile = {}
        z = open('input.txt', 'rt')
        for line in z.readlines():
                cols = line.split()
                inputFile[cols[0]] = cols[1]
                z.close()
        main_m = []
        for i in range(len(img_dict)):
	        main_m.append([0]*len(img_dict))

        def float_format(vector, decimal):
                return np.round((vector).astype(np.float), decimals=decimal)
 
        dp = 1/len(inputFile)

        for i in graph_list:
                curr_img = i[0]
                curr_sum = 0
                for k in i[1]:
                        curr_sum+=k[1]
                for k in i[1]:
                        main_m[img_dict[curr_img]][img_dict[k[0]]] = k[1]/curr_sum

        M = np.transpose(np.matrix(main_m))
        E = np.zeros((1, int(len(img_dict))))
        for i in inputFile:
                E[0][img_dict[int(i)]] = dp

        c = 0.15

        uq = np.matrix(E)
        prev_uq = uq

        for i in range(100):
                uq = ((1-c) * uq * M + (c * E))
                if(prev_uq == uq).all():
                        break
                prev_uq = uq
        uq = uq.tolist()[0]
        ids = list(img_dict.keys())

        id_scores = []
        for i in range(len(uq)):
                id_scores.append((ids[i], uq[i]))

        id_scores = sorted(id_scores, key= lambda x:x[1], reverse=True)

        label_dict = {}
        dict_by_label = {}
        for i in id_scores:
                min = 1.0
                for j in inputFile:
                        j_index = [y[0] for y in id_scores].index(int(j))
                        if(abs(i[1] - id_scores[j_index][1]) < min):
                                min = abs(i[1] - id_scores[j_index][1])
                                min_id = j
                label_dict[i[0]] = inputFile[min_id]

        for i in set(label_dict.values()):
                dict_by_label[i] = []

        for i in label_dict:
                dict_by_label[label_dict[i]].append(i)

        file = open("../preprocessedData/img_loc_dict","rb")
        mapping = pickle.load(file)
        file.close()

        label_img_dict = {}
        for label in dict_by_label:
                img_list = list()
                f = open('Task6_PPR_'+label+'.html','a+')
                for image in dict_by_label[label]:
                        img_location = mapping[image]
                        file_to_show = "../img/"+str(img_location)+"/"+str(image)+".jpg"
                        img_list.append(file_to_show)
                        message = '''\
                        ... <html> <body> <div> {label}  <img src={img}> </img> </div></body>.\
                        ... <html>'''.format(img=file_to_show, label=str(image)+ "  " +str(img_location))
                        f.write(message)
                label_img_dict[label] = img_list
                f.close()
