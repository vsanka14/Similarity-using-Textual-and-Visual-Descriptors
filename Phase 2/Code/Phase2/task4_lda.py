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
import scipy
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

def task_lda(visual_descriptor_model,no_topics,given_location_id):
   
   locations_dict = get_location_names()    ##RETURN A LOCATION ID TO NAME MAPPING.
   location_names = locations_dict.values()
   df = csv_parser.parse_csv(location_names,visual_descriptor_model)
   
   locations_from_df = df["location_name"]
 #  print(type(locations_from_df))

   df.drop(columns="location_name",inplace=True,axis=1)
   base_location = locations_dict[given_location_id]
#   print(len(df))
   scaler = MinMaxScaler()
   df = scaler.fit_transform(df)
#   print("After scaling : \n")
#   print(df)
   
##IDEA. FOR THIS IS THAT WE NEED TO CONSTRUCT A DATAFRAME OUT OF OUR GIVEN LOCATION + ANOTHER LOCATION.
##THEN DO A CDIST.

   print("Training model...")
   lda_model = LatentDirichletAllocation(no_topics, max_iter=10, learning_offset=50., random_state=0).fit(df)
   lda_W = lda_model.transform(df)  ##image->topic
   lda_H = lda_model.components_    ##topic->feature
   image_feature_df = pd.DataFrame(lda_W)  #this is a topic->feature matrix
#	df = pd.DataFrame(np.sort(df.values, axis=1), index=df.index, columns=df.columns)

   topic_feature_df = pd.DataFrame(lda_H)
   top_k_features = len(topic_feature_df.columns)
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
	  
   image_feature_df['location_name'] = locations_from_df.values
#   print(len(image_feature_df))
#   print(image_feature_df.head())
   
#   print(image_feature_df.head())
   base_location_df = image_feature_df[image_feature_df["location_name"] == base_location]
   base_location_df.drop(columns="location_name",axis=1,inplace=True)
   
#   print(base_location_df.head())
   final_distances_list = []
  
   for loc in location_names:
      dist = 0
      if(loc == base_location):
          continue
      compared_location_df = image_feature_df[image_feature_df["location_name"] == loc]
      compared_location_df.drop(columns="location_name",axis=1,inplace=True)
#     print(compared_location_df.head())
      dist_matrix = scipy.spatial.distance.cdist(base_location_df,compared_location_df,metric='euclidean')
      for row in dist_matrix:
         min_dist = min(row)
         dist += min_dist
      final_distances_list.append((loc,dist))		  
   final_distances_list = sorted(final_distances_list,key=operator.itemgetter(1))    ##sort in ascending based on distance   
#   print(final_distances_list)
   
   k = 5
   print("\n\nThe top 5 similar locations are : \n")
   for place,dist in final_distances_list[:k]:
      print("Location Name : " , place , " Similarity : " , dist)	  

   
def main():
	visual_descriptor_model = input("Please enter the visual descriptor model (CM/CM3x3/CN/CN3x3/CSD/GLRLM/GLRLM3x3/HOG/LBP/LBP3x3): ")
	k = int(input("Please enter the value of k:"))
	location_id = input("Please enter location id : ")
	
	#all_models = ["CM.csv", "CM3x3.csv", "CN.csv", "CN3x3.csv", "CSD.csv", "GLRLM.csv" , "GLRLM3x3.csv", "HOG.csv", "LBP.csv", "LBP3x3.csv"]
#	print(locations)
#	print(df.head())
	task_lda(visual_descriptor_model,k,location_id)
	
if __name__ == '__main__':
	main()