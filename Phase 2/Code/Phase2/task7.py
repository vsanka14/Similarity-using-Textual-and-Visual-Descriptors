import parsers as parser
import pandas as pd
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

def getTensor():
    #userData
    userDict = parser.parseUsersFile()
    #convert to form of user-set(words)
    userDict = parser.getWordSet(userDict)

    #imageData
    imageDict = parser.parseImageFile()
    #convert to form of image-set(words)
    imageDict = parser.getWordSet(imageDict)


    #locationData
    locationDict = parser.parseLocationFile()
    #convert to form of location-set(words)
    locationDict = parser.getWordSet(locationDict)

    tensor = [[[0 for k in range(len(locationDict))] for j in range(len(imageDict))] for i in range(len(userDict))]

    userIndex = 0
    for user in userDict:
        imageIndex = 0
        for image in imageDict:
            locationIndex = 0
            for location in locationDict:
                tensor[userIndex][imageIndex][locationIndex] = len(userDict[user].intersection(imageDict[image], locationDict[location]))
                locationIndex += 1
            imageIndex += 1
        userIndex += 1
    tensor = np.array(tensor, dtype='float64')
    return tensor


def task():
    k = input("Enter CP-Rank: ")
    tensor = getTensor()
    factors = parafac(tensor, rank = int(k))
    print(factors)
    print([factor.shape for factor in factors])
    # print(len(factors))
