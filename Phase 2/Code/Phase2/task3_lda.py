import csv_parsers_siddharth as csv_parser
import sys
import os
import xml.etree.ElementTree as ET
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import scipy
from scipy.spatial import distance
import numpy as np
import operator
from sklearn.preprocessing import MinMaxScaler

def get_location_names():
	loc_names = {}
	tree = ET.parse("devset_topics.xml")
	root = tree.getroot()
	for topic in root.findall('topic'):
		loc_name = topic.find('title').text
		loc_id = topic.find('number').text
		loc_names.update({loc_id : loc_name})
	return loc_names

def task_lda(visual_descriptor_model,no_topics,given_item):

   print("Constructing dataframe..")
   location_dict = get_location_names()   ##gets all location names.
   locations = location_dict.values()
   df = csv_parser.parse_csv(locations,visual_descriptor_model)
  
   print(df.head())
   item_names = df.index.tolist()      ##get the image ids
   df.reset_index(drop=True,inplace=True)   #reset the index
 
   locations = df["location_name"].tolist() #get the corresponding locations  
   df.drop(["location_name"],axis=1,inplace=True)  #dropl location name column so that there are no errors in distance computation
   
   scaler = MinMaxScaler()
   df = scaler.fit_transform(df)
   
   img_to_location_dict = {}
   for item,loc in zip(item_names,locations):
     img_to_location_dict.update({item : loc}) 		##print the item->location mapping
  
#	print(lda_model.components_)

   print("Training model...")
   lda_model = LatentDirichletAllocation(no_topics, max_iter=10, learning_offset=50., random_state=0).fit(df)
   lda_W = lda_model.transform(df)  ##image->topic
   lda_H = lda_model.components_    ##topic->feature
   image_feature_df = pd.DataFrame(lda_W)  #this is a image->topic matrix
   topic_feature_df = pd.DataFrame(lda_H)  #this is a topic->feature matrix
#	df = pd.DataFrame(np.sort(df.values, axis=1), index=df.index, columns=df.columns)


   print(topic_feature_df.head())
   top_k_features = len(topic_feature_df.columns)
#   print(type(topic_feature_df))
   
   print("Model training completed..")
   
   feature_weight_list = []
   for index,row in topic_feature_df.iterrows(): ##go through each of the topics.
      feature_weights = row.values        ##get weight of each feature.(the columns)
      for i in range(0,len(feature_weights)):
        feature_weight_list.append((i,feature_weights[i]))
      feature_weight_list = sorted(feature_weight_list,key=operator.itemgetter(1),reverse=True)
      feature_weight_list = feature_weight_list[:top_k_features]
      print("\nFeature weights for topic " , str(index+1) , " are : ")
      print(feature_weight_list)
      feature_weight_list.clear()
   
   print("Starting similarity computations...")
   
   given_item_idx = 0
   for i in range(0,len(item_names)):
     if(item_names[i] == given_item):
       given_item_idx = i
       break  
	   
   final_distances_list = []
   given_item_ndarray = image_feature_df.iloc[given_item_idx].values          ##get the given image values.
   
 #  print("\nGiven item ndarray : ")
 #  print(given_item_ndarray)
 #  print("\n size : " , given_item_ndarray.size)
   
   count = 0
   for index,row in image_feature_df.iterrows(): 
      if(index == given_item_idx):
         continue
      curr_item_ndarray = row.values
      dist = np.linalg.norm(given_item_ndarray-curr_item_ndarray)
      final_distances_list.append((item_names[index],dist))
   
   k = 5
   length = len(final_distances_list) 
   closest_k_items = sorted(final_distances_list,key = operator.itemgetter(1))
   
   print("\nThe 5 closest images to the given image : \n")
   for img,dist in closest_k_items[:k]:
      print("{ img id : " , img , " distance : " , dist , "}")
 
   related_locations = 5     ##parameter to control no of locations
   seen = 0
   seen_locations = set()
   print("\n\nThe 5 closest locations to the given image : \n")
   
   for img,dist in closest_k_items:
      if(img_to_location_dict[img] in seen_locations):
          continue
      seen_locations.add(img_to_location_dict[img])
      print(" location id : " , img_to_location_dict[img] , " dist : " , dist , "}")
      seen += 1
      if(seen == related_locations):
         break
	#tfidf_feature_names = count_vectorizer.get_feature_names()
   	
def main():
	visual_descriptor_model = input("Please enter the visual descriptor model (CM/CM3x3/CN/CN3x3/CSD/GLRLM/GLRLM3x3/HOG/LBP/LBP3x3): ")
	k = int(input("Please enter the value of k:"))
	image_id = input("Please enter image id : ")
	
	#all_models = ["CM.csv", "CM3x3.csv", "CN.csv", "CN3x3.csv", "CSD.csv", "GLRLM.csv" , "GLRLM3x3.csv", "HOG.csv", "LBP.csv", "LBP3x3.csv"]
	
#	print(locations)
#	print(df.columns)
#	print(df.head())
	task_lda(visual_descriptor_model,k,image_id)
	
if __name__ == '__main__':
	main()