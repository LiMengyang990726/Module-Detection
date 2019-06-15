import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit

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
def split(data1,data2,data3):
    # StratifiedShuffleSplit for data1
    split = StratifiedShuffleSplit(test_size=0.3, random_state=42)
    for train_index, test_index in split.split(data1,data1['Target']):
        strat_train_set_1 = data1.loc[train_index]
        strat_test_set_1 = data1.loc[test_index]
    #
    # StratifiedShuffleSplit for data1
    split = StratifiedShuffleSplit(test_size=0.3, random_state=43)
    for train_index, test_index in split.split(data2,data2['Target']):
        strat_train_set_2 = data2.loc[train_index]
        strat_test_set_2 = data2.loc[test_index]
    #
    # StratifiedShuffleSplit for data1
    split = StratifiedShuffleSplit(test_size=0.3, random_state=44)
    for train_index, test_index in split.split(data3,data3['Target']):
        strat_train_set_3 = data3.loc[train_index]
        strat_test_set_3 = data3.loc[test_index]
    #
    return (strat_train_set_1,strat_test_set_1,strat_train_set_2,strat_test_set_2,strat_train_set_3,strat_test_set_3)


def balance(data):
    data_dis = data[data['Target']==1]
    data_nondis = data[data['Target']==0]
    data1_nondis = data_nondis.sample(n = 1441, random_state = 1)    # known that the best param will be 1441 for keeping it balanced
    data1 = pd.concat([data1_nondis,data_dis])
    #
    temp = data_nondis.drop(data1_nondis.index)
    data2_nondis = temp.sample(n = 1441, random_state = 2)
    data2 = pd.concat([data2_nondis,data_dis])
    #
    data3_nondis = temp.drop(data2_nondis.index)
    data3 = pd.concat([data3_nondis,data_dis])
    #
    data1.reset_index(inplace=True)
    data1.drop(['index'],axis = 1, inplace = True)
    data2.reset_index(inplace=True)
    data2.drop(['index'],axis = 1, inplace = True)
    data3.reset_index(inplace=True)
    data3.drop(['index'],axis=1, inplace = True)
    return (data1, data2, data3)

def output():
    # path = "/home2/lime0400/Module-Detection/"                        # If in the server
    path = "/Users/limengyang/Workspaces/Module-Detection/"
    data = readInput(path)
    (data1, data2, data3) = balance(data)
    return split(data1, data2, data3)