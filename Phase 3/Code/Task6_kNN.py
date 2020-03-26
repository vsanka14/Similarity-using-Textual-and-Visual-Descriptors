'''
Implementation of k-nearest neighbour based classification algorithm.
Code developed in Python 3.7.0
'''
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle

def task():
    print("Enter value of 'k' for k-Nearest Neighbours: ")
    k = int(input())
    
    # Read the input file containing a set of image/label pairs.
    # Store these values in a dictionary with "image id" as "key" and "image label" as "value".
    inputFile = {}
    dict_by_label = {}
    z = open('input.txt', 'rt')
    for line in z.readlines():
        cols = line.split()
        inputFile[cols[0]] = cols[1]
    z.close()

    for i in set(inputFile.values()):
        dict_by_label[i] = []

    # Read the image dataset file and store in a dictionary of dictionaries defined as:
    # image_dict = {"image_id" : {"term" : "tf-idf"}}
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path = '../desctxt/devset_textTermsPerImage.txt'
    f = open(os.path.join(fileDir,path), "r", encoding = 'utf8')
    lines = f.readlines()
    image_dict = {}
    for line in lines:
        line_arr = line.split(" ")
        image_dict[line_arr[0]] = {}
        for i in range(1, len(line_arr) - 1, 4):
            image_dict[line_arr[0]][line_arr[i].replace('"', '')] = line_arr[i+3]

    # All terms of the images contained in the input file are stored in a dictionary.
    inputFileTerms = {}
    for keys in image_dict:
        if keys in inputFile:
            inputFileTerms[keys] = image_dict[keys]
    
    # Dictionary for keeping count of images classified as having a particular image label.
    # Count for any new image label added to the dictionary, will be initialized with zero.
    label_counts = Counter()

    # Calculate the cosine similarity scores of each image in the dataset with respect to the images in input file.
    # Select the top 'k' images on basis of higher cosine similarity score.
    for keys in image_dict:
        sorted_label = []
        cosine_dist = Counter()
        for inp in inputFileTerms:
            for term in image_dict[keys]:
                if term in inputFileTerms[inp]:
                    cosine_dist[inp] += np.dot(float(image_dict[keys][term]), float(image_dict[inp][term]))
                else:
                   cosine_dist[inp] += 0.0
        cosine_dist = sorted(cosine_dist.items(), key = lambda x:x[1], reverse = True)[:k]
        
        # Get image label of the top 'k' similar images from input file.
        for i in cosine_dist:
            sorted_label.append(inputFile[i[0]])
        del(cosine_dist)
        
        # Count the number of images with a particular image label among 'k' neighbouring images.
        # The image in dataset is assigned the image label with the most number of neighbours.
        sorted_label_count = Counter()
        for i in sorted_label:
            sorted_label_count[i]+=1
        sorted_label_count = sorted(sorted_label_count.items(), key = lambda x:x[1], reverse= True)
        
        label_counts[sorted_label_count[0][0]] += 1
        dict_by_label[sorted_label_count[0][0]].append(int(keys))
        
        del(sorted_label_count)
        del(sorted_label)
    
    # Command line output with image label and number of images assigned with the label.
    # Plot a bar graph with image label as X-axis and Number of images assigned with the label as Y-axis.
    print(len(lines), "images classified using", k, "-Nearest Neighbour classification.")
    for i in label_counts:
        print("Number of images assigned with label", i, " : ", int(label_counts[i]))
        plt.bar(i, int(label_counts[i]))
    plt.xlabel("Image Labels")
    plt.ylabel("Number of Images")
    plt.title(str(len(lines))+" images labeled using "+ str(k) +"-NN")
    plt.show()

    file = open("./preprocessedData/img_loc_dict","rb")
    mapping = pickle.load(file)
    file.close()

    label_img_dict = {}
    for label in dict_by_label:
        img_list = list()
        f = open('Task6_kNN_'+label+'.html','a+')
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
