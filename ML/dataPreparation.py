import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import os

##################################
# Data preparation
##################################

######### Read input files
def readInput(path):
    absolutePath = path
    data = pd.read_csv(os.path.join(absolutePath,"cleanFeatures.csv"))
    data.drop(['Unnamed: 0'], axis=1,inplace = True)
    return data


######### Split train test set
def split(data):
    X = data.drop(['Target'],axis=1)
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2019) # change to 0.3
    return (X_train, X_test, y_train, y_test)

def output():
    # path = "/home2/lime0400/Module-Detection/"                        # If in the server
    path = "/Users/limengyang/Workspaces/Module-Detection/"
    data = readInput(path)
    return split(data)
