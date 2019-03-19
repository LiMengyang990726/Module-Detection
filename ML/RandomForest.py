import pandas as pd
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline
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

##################################
# Method 3: Random Forest
##################################
def RandomForest(X_train_RF,X_test_RF, y_train, y_test):
    ######## Prepare output
    text_file = open("RandomForestOutput.txt", "w")
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
    rf_random_f1 = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                            scoring = "f1", n_iter = 100, cv = 3, verbose=2,
                            random_state=42, n_jobs = -1)
    rf_random_acc = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                            scoring = "accuracy", n_iter = 100, cv = 3, verbose=2,
                            random_state=42, n_jobs = -1)
    rf_random_f1.fit(X_train_RF, y_train)
    rf_random_acc.fit(X_train_RF, y_train)
    #
    ######### Evaluate Result
    text_file.write("The best f1 score for randomizedsearchcv is " + str(rf_random_f1.best_score_) + "\n")
    text_file.write("The best acc score for randomizedsearchcv is " + str(rf_random_acc.best_score_) + "\n")
    #
    f1_params = rf_random_f1.best_params_
    acc_params = rf_random_acc.best_params_
    #
    ######### Make Predictions
    rf_f1 = RandomForestClassifier(n_estimators = f1_params['n_estimators'],
                                    max_features = f1_params['max_features'],
                                    max_depth = f1_params['max_depth'],
                                    min_samples_split = f1_params['min_samples_split'],
                                    min_samples_leaf = f1_params['min_samples_leaf'],
                                    bootstrap = f1_params['bootstrap'])
    rf_f1.fit(X_train_RF, y_train)
    predictions = rf_f1.predict(X_test_RF)
    actual = y_test.tolist()
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(predictions)):
        if((predictions[i] == 0) and (actual[i] == 1)):    # false negative: predicted negative (non disease is 0), actual positive (disease is 1)
            FN += 1
        if((predictions[i] == 0) and (actual[i] == 0)):    # true negative: predicted negative (non disease is 0), actual negative (non disease is 0)
            TN += 1
        if((predictions[i] == 1) and (actual[i] == 1)):    # true positive: predicted positive (disease is 1), actual positive (disease is 1)
            TP += 1
        if((predictions[i] == 1) and (actual[i] == 0)):    # false positive: predicted positive (disease is 1), actual negative (non disease is 0)
            FP += 1

    #
    F1 = (2*TP)/(2*TP+FP+FN)
    text_file.write("F1 score for the test set at the best parameter is " + str(F1) + "\n") #0.8878143133462283
    #
    rf_acc = RandomForestClassifier(n_estimators = acc_params['n_estimators'],
                                    max_features = acc_params['max_features'],
                                    max_depth = acc_params['max_depth'],
                                    min_samples_split = acc_params['min_samples_split'],
                                    min_samples_leaf = acc_params['min_samples_leaf'],
                                    bootstrap = acc_params['bootstrap'])
    rf_acc.fit(X_train_RF, y_train)
    predictions = rf_acc.predict(X_test_RF)
    actual = y_test.tolist()
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(predictions)):
        if((predictions[i] == 0) and (actual[i] == 1)):    # false negative: predicted negative (non disease is 0), actual positive (disease is 1)
            FN += 1
        if((predictions[i] == 0) and (actual[i] == 0)):    # true negative: predicted negative (non disease is 0), actual negative (non disease is 0)
            TN += 1
        if((predictions[i] == 1) and (actual[i] == 1)):    # true positive: predicted positive (disease is 1), actual positive (disease is 1)
            TP += 1
        if((predictions[i] == 1) and (actual[i] == 0)):    # false positive: predicted positive (disease is 1), actual negative (non disease is 0)
            FP += 1

    #
    ACC = (TP+TN) / len(predictions)
    text_file.write("Acc score for the test set at the best parameter is " + str(ACC) + "\n")
