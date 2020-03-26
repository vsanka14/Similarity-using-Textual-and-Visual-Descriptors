# -*- coding: utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
import os

def task():


    #Read input <<k>>  format
    inp = input("Enter <<k img1 img2 img3>> :")
    inpStringArray = inp.split()

    vector_space = inpStringArray[0]
    model = inpStringArray[1]
    k_value = 0
    try :
         k_value = int(inpStringArray[0])
         imgs = [int(i) for i in inpStringArray[1:]]
    except:
        print("Error : Please enter valid value of k ")
        return




    file = open("./preprocessedData/img_sim_graph","rb")
    graph = pickle.load(file)
    print("Read data from pickle")
    file.close()
    
    file = open("./preprocessedData/img_dict","rb")
    img_dict = pickle.load(file)
    print("Read img_dict from pickle")
    file.close()
    
    graph_list = graph.items()
    len_graph = len(graph_list)
    main_m = []
    for i in range(len_graph):
    	main_m.append([0]*len(graph_list))
    
    
    def float_format(vector, decimal):
        return np.round((vector).astype(np.float), decimals=decimal)
     
    dp = 1/len(imgs)
    
    for i in graph_list:
            curr_img = i[0]
            curr_sum = 0
            for k in i[1]:
                    curr_sum+=1 - k[1]
            for k in i[1]:
                    main_m[img_dict[curr_img]][img_dict[k[0]]] = (1- k[1])/curr_sum
    #print(main_m)                        
    
            
    # WWW matrix
    #Here M is adj list with weights
    M = np.transpose(np.matrix(main_m))
     
    E = np.zeros((1,len(graph_list)))
    for img in imgs :
        E[0][img_dict[img]] = dp

    #E[:] = dp
     
    # taxation
    beta = 0.85
    uq = np.matrix(E)
    prev_uq = uq

    for i in range(100):
    # WWW matrix
        uq = beta *uq* M + ((1-beta) * E)
        prev_uq = uq
    uq = uq.tolist()[0]
#    print(uq)

# print("A: ", A)
    
    # initial vector
    
#    r = np.matrix([dp]*len(graph_list))
#    r = np.transpose(r)

#    previous_r = r
#    for it in range(1,5000):
#        r = A * r
#
#        #check if converged
#        if (previous_r==r).all():
#            # print(it)
#            break
#        previous_r = r
#    # r = sorted(r, reverse=True)
    # print(r)
    # print ("Final:\n", float_format(r,3))
    # print(np.argmax(r))

    k=k_value



#    k_arr = np.array(list(itertools.chain(*(r.tolist()))))
    k_arr = np.array(uq)
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
        os.remove('task4.html')
    except OSError:
        pass
    f = open('task4.html','a+')
    for tup in img_srcs:
        message = '''\
        ... <html> <body> <div> {label}  <img src={img}> </img> </div></body>.\
        ... <html>'''.format(img=tup[0], label=str(tup[1])+ "  " +str(tup[2]))
        f.write(message)
    f.close()
