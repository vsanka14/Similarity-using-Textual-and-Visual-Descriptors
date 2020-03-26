# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
import os

def task():


    #Read input <<k>>  format
    inp = input("Enter <<k>> :")
    k_value = 0
    try :
         k_value = int(inp)
    except:
        print("Error : Please enter valid value of k ")
        return




    file = open("./preprocessedData/img_sim_graph","rb")
    graph = pickle.load(file)
    # print("Read data from pickle")
    file.close()
    
    file = open("./preprocessedData/img_dict","rb")
    img_dict = pickle.load(file)
    # print("Read img_dict from pickle")
    file.close()
    
    graph_list = graph.items()
        
    main_m = []
    for i in range(len(graph_list)):
    	main_m.append([0]*len(graph_list))
    
    
    def float_format(vector, decimal):
        return np.round((vector).astype(np.float), decimals=decimal)
     
    dp = 1/len(graph_list)
    
    for i in graph_list:
            curr_img = i[0]
            curr_sum = 0
            for k in i[1]:
                    curr_sum+=1 - k[1]
            for k in i[1]:
                    main_m[img_dict[curr_img]][img_dict[k[0]]] = (1- k[1])/curr_sum
    #print(main_m)                        
    
            

    M = np.transpose(np.matrix(main_m))
     
    E = np.zeros((len(graph_list),len(graph_list)))
    E[:] = dp
     
    
    bet = 0.85
     

    A = bet * M + ((1-bet) * E)
     

    r = np.matrix([dp]*len(graph_list))
    r = np.transpose(r)
     
    previous_r = r
    for it in range(1,5000):
        r = A * r
        
  
        if (previous_r==r).all():
            # print(it)        
            break
        previous_r = r
    # r = sorted(r, reverse=True) 
    # print(r)
    # print ("Final:\n", float_format(r,3))
    # print(np.argmax(r))
    k=k_value



    k_arr = np.array(list(itertools.chain(*(r.tolist()))))
    top_k = k_arr.argsort()[-k:][::-1]
    # print(top_k)
    img_srcs = list()
    inv_map = {v: k for k, v in img_dict.items()}
    for i in top_k:
        max_id = i
        img = inv_map[max_id]
        
        file = open("./preprocessedData/img_loc_dict","rb")
        mapping = pickle.load(file)
        # print("Read data from pickle")
        file.close()
        img_location = mapping[img]
        file_to_show = "../img/"+str(img_location)+"/"+str(img)+".jpg"
        # print(file_to_show)
        # print ("sum", np.sum(r))
        img_srcs.append((file_to_show,img,img_location))
        print(img," \t", img_location, "\t", k_arr[i])
        # image = mpimg.imread(file_to_show)
        # imgplot = plt.imshow(image)
        # plt.show()
    
    try:
        os.remove('task3.html')
    except OSError:
        pass
    f = open('task3.html','a+')
    for tup in img_srcs:
        message = '''\
        ... <html> <body> <div> {label}  <img src={img}> </img> </div></body>.\
        ... <html>'''.format(img=tup[0], label=str(tup[1]) + "  " +str(tup[2]))
        f.write(message)
    f.close()
