import pandas as pd
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import operator
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn import preprocessing
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import pickle
import ConfusionMatrix

##################################
# Method 3: Random Forest
##################################
def randomForest(X_train_RF,y_train, number):
    #
    ######### Create model
    rf = RandomForestClassifier()
    #
    ######### Search for the best parameter
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(5, 105, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                            n_iter = 100, cv = 3, verbose=2,
                            random_state=42, n_jobs = -1)
    rf_random.fit(X_train_RF, y_train)
    # #
    # ######### Record the best parameters
    # #
    # params = rf_random.best_params_
    # #
    # ######### Make Predictions
    # rf_good = RandomForestClassifier(n_estimators = params['n_estimators'],
    #                                 max_features = params['max_features'],
    #                                 max_depth = params['max_depth'],
    #                                 min_samples_split = params['min_samples_split'],
    #                                 min_samples_leaf = params['min_samples_leaf'],
    #                                 bootstrap = params['bootstrap'])
    # rf_good.fit(X_train_RF, y_train)
    #
    # On the Cross Validaton Set
    result = cross_val_score(rf_random, X_train_RF, y_train,cv=10)
    output = pd.DataFrame({'CV1':[],'CV2':[],'CV3':[],'CV4':[],
                        'CV5':[],'CV6':[],'CV7':[],
                        'CV8':[],'CV9':[],'CV10':[]})
    output = output.append({
            'CV1':result[0],'CV2':result[1],'CV3':result[2], 'CV4':result[3],
            'CV5':result[4],'CV6':result[5],'CV7':result[6],
            'CV8':result[7],'CV9':result[8],'CV10':result[9]
    }, ignore_index=True)
    output.to_csv("RandomForest_"+str(number)+".csv")
    #
    # Save model
    pkl_filename = "random_forest_model_"+str(number)+".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(rf_random, file)

def evaluation(X_test_SVM,y_test,number):
    # Load Model
    pkl_filename = "random_forest_model_"+str(number)+".pkl"
    with open(pkl_filename, 'rb') as file:
        rf = pickle.load(file)
    #
    # Make Prediction
    result = rf.predict(X_test_SVM)
    #
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(result)):
        if((result[i] == 0) and (y_test[i] == 1)):    # false negative: predicted negative (non disease is 0), actual positive (disease is 1)
            FN += 1
        if((result[i] == 0) and (y_test[i] == 0)):    # true negative: predicted negative (non disease is 0), actual negative (non disease is 0)
            TN += 1
        if((result[i] == 1) and (y_test[i] == 1)):    # true positive: predicted positive (disease is 1), actual positive (disease is 1)
            TP += 1
        if((result[i] == 1) and (y_test[i] == 0)):    # false positive: predicted positive (disease is 1), actual negative (non disease is 0)
            FP += 1
    #
    output = pd.DataFrame({'Cond':[],
                  'TP': [],'FN': [],'FP': [],'TN':[],
                  'Accuracy':[],'Precision':[],'Recall':[],'F1 score':[]})
    result = ConfusionMatrix.confusionMatrix(TP,FN,FP,TN)
    cond = "data set "+str(number)
    output = output.append({'Cond':cond,
                  'TP': TP,'FN': FN,'FP': FP,'TN':TN,
                  'Accuracy':result[0],'Precision':result[1],'Recall':result[2],'F1 score':result[3]},
                  ignore_index=True)
    output.to_csv("RandomForestOutput_"+str(number)+".csv")
