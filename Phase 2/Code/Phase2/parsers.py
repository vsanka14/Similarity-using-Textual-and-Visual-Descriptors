import os
from collections import defaultdict
import xml.etree.ElementTree as ET
import pandas as pd

fileDir = os.path.dirname(os.path.realpath('__file__'))

#Function to parse users from a file
def parseUsersFile():

  filePath = '../desctxt/devset_textTermsPerUser.txt'
  fileHandler = open(os.path.join(fileDir, filePath), "r", encoding = 'utf8')
  lines = fileHandler.readlines()
  userDictionary = defaultdict(dict)
  for line in lines:
    line = line.strip()
    lineArray = line.split(" ")
    userDictionary[lineArray[0]] = defaultdict(dict)
    for i in range(1, len(lineArray) - 1, 4):
      userDictionary[lineArray[0]][lineArray[i].replace('"', '')] = [lineArray[i+1], lineArray[i+2], lineArray[i+3]]
  return userDictionary


#Function to parse images from a file
def parseImageFile():
  filePath = '../desctxt/devset_textTermsPerImage.txt'
  fileHandler = open(os.path.join(fileDir, filePath), "r", encoding = 'utf8')
  lines = fileHandler.readlines()
  imageDictionary = defaultdict(dict)
  for line in lines:
    lineArray = line.split(" ")
    line = line.strip()
    imageDictionary[lineArray[0]] = defaultdict(dict)
    for i in range(1, len(lineArray) - 1, 4):
      imageDictionary[lineArray[0]][lineArray[i].replace('"', '')] = [lineArray[i+1], lineArray[i+2], lineArray[i+3]]
  return imageDictionary

#Function to parse locations from a file
def parseLocationFile():
  locationDictionary = defaultdict(dict)
  tree = ET.parse('../devset_topics.xml')
  topics = tree.findall('topic')
  for topic in topics:
    locationDictionary[topic.find('title').text] = defaultdict(dict)
    locationDictionary[topic.find('title').text]['id'] = topic.find('number').text

  filePath = '../desctxt/devset_textTermsPerPOI.wFolderNames.txt'
  f = open(os.path.join(fileDir, filePath), "r", encoding = 'utf8')
  lines = f.readlines()
  for line in lines:
    line = line.strip()
    lineArray = line.split(" ")
    for item in lineArray[0].split("_"):
      lineArray.remove(item)
    for i in range(1, len(lineArray) - 1, 4):
      locationDictionary[lineArray[0]][lineArray[i].replace('"', '')] = [lineArray[i+1], lineArray[i+2], lineArray[i+3]]
  return locationDictionary


def processMapToDataFrame(itemDictionary, index):
    
    '''Map structure  Users = { usr1 :{ "wd": { tf : 1, df: 2 , idf: 1 }} ,
                                usr2 :{ "wd2": { tf : 1, df: 2 , idf: 1 , "wd3": { tf : 1, df: 2 , idf: 1 }}
                                }
    '''
    itemList = list()
    indexNameList = list()
    for item,itemVal in itemDictionary.items() :   # item -> user
        indexNameList.append(item)
        processedMap = defaultdict()
        for key, val in itemVal.items() : #item_word -> word
            processedMap[key] = int(val[index])    
        itemList.append(processedMap)
    return (itemList,indexNameList)

def processMapToDataFrameAll(userDictionary, imageDictionary, locationDictionary, index):
    
    '''Map structure  Users = { usr1 :{ "wd": { tf : 1, df: 2 , idf: 1 }} ,
                                usr2 :{ "wd2": { tf : 1, df: 2 , idf: 1 , "wd3": { tf : 1, df: 2 , idf: 1 }}
                                }
    '''
    itemList = list()
    indexNameList = list()
    indexImageList = list()
    indexLocList = list()
    
    for item,itemVal in userDictionary.items() :   # item -> user
        indexNameList.append(item)
        processedMap = defaultdict()
        for key, val in itemVal.items() : #item_word -> word
            processedMap[key] = int(val[index])    
        itemList.append(processedMap)

    for item,itemVal in imageDictionary.items() :   # item -> user
        indexImageList.append(item)
        processedMap = defaultdict()
        for key, val in itemVal.items() : #item_word -> word
            processedMap[key] = int(val[index])    
        itemList.append(processedMap)
    
    for item,itemVal in locationDictionary.items() :   # item -> user
        indexLocList.append(item)
        processedMap = defaultdict()
        for key, val in itemVal.items() : #item_word -> word
            processedMap[key] = int(val[index])    
        itemList.append(processedMap)

    
    return (itemList,indexNameList,indexImageList,indexLocList)

    
def parse_csv(location_dict, model):
    all_data = []
    for location in location_dict:
        df = pd.read_csv("../descvis/img/" + location+ " " + model + ".csv", header = None) 
        df['location'] = location
        df.rename(columns={0: 'img_id'}, inplace=True) 
        df.set_index('img_id', inplace=True, drop=True)
        all_data.append(df)
    csv_df = pd.concat(all_data)
    return csv_df

def parse_csv_with_location(location_dict, model):
    all_data = []
    for location in location_dict:
        df = pd.read_csv("../descvis/img/" + location+ " " + model + ".csv", header = None) 
        df['location'] = location
        df.rename(columns={0: 'img_id'}, inplace=True) 
        df.set_index(['img_id','location'], inplace=True, drop=True)
        all_data.append(df)
    csv_df = pd.concat(all_data)
    return csv_df

def parse_all_location_models(location_dict):
    all_data = []
    models = ['CM', 'CM3x3', 'CN', 'CN3x3', 'CSD', 'GLRLM', 'GLRLM3x3', 'HOG', 'LBP', 'LBP3x3']
    for model in models:
      for location in location_dict:
          df = pd.read_csv("../descvis/img/" + location + " " + model + ".csv", header = None) 
          df['location'] = location
          df['model'] = model
          df.rename(columns={0: 'img_id'}, inplace=True) 
          df.set_index(['img_id', 'location', 'model'], inplace=True, drop=True)
          all_data.append(df.sample(100))
    csv_df = pd.concat(all_data)
    csv_df = csv_df.fillna(0)
    return csv_df

def getWordSet(itemDict):
    ans = defaultdict(set)
    for item in itemDict:
        for word in itemDict[item]:
            ans[item].add(word)
    return ans

def processMapToDataFrame_NEW(data,index):
    
    '''Map structure  Users = { usr1 :{ "wd": { tf : 1, df: 2 , idf: 1 }} ,
                                usr2 :{ "wd2": { tf : 1, df: 2 , idf: 1 , "wd3": { tf : 1, df: 2 , idf: 1 }}
                                }
    '''
    itemList = list()
    itemIndexMap = defaultdict()

    print(len(data))
    i = 0
    for item,itemVal in data.items() :   # item -> user
        itemIndexMap[item] = i
        i += 1
        processedMap = defaultdict()
        for key, val in itemVal.items() : #item_word -> word
            processedMap[key] = int(val[index])    
        itemList.append(processedMap)    
    print(len(itemList[0]), len(itemIndexMap))
    return (itemList,itemIndexMap)