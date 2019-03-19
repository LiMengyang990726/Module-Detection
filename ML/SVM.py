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
# Method 2: Support Vector machine with nested and non-nested cross validation
##################################

######### Solve the error of "Continuous 0"
def encodeY(y_train,y_test):
    lab_enc = preprocessing.LabelEncoder()
    y_encoded_train = lab_enc.fit_transform(y_train)
    print(y_encoded_train)
    print(utils.multiclass.type_of_target(y_train))
    print(utils.multiclass.type_of_target(y_train.astype('int')))
    print(utils.multiclass.type_of_target(y_encoded_train))
    #
    y_encoded_test = lab_enc.fit_transform(y_test)
    print(y_encoded_test)
    print(utils.multiclass.type_of_target(y_test))
    print(utils.multiclass.type_of_target(y_test.astype('int')))
    print(utils.multiclass.type_of_target(y_encoded_test))
    #
    return (y_encoded_train, y_encoded_test)

######### Linear Kernel
def SVMKernelLinear(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test):
    ######### Prepare output
    text_file = open("SVMLinear.txt", "w")
    #
    #########  Prepare parameters
    NUM_TRIALS = 5
    p_grid = {"C": [1, 10, 100], "gamma": [0.001,.01, .1]}
    svm = SVC(kernel='linear')
    #
    ######### Score record
    nested_f1 = np.zeros(NUM_TRIALS)
    nested_acc = np.zeros(NUM_TRIALS)
    #
    ######### Find the best parameters from the non-nested cross validation
    inner_cv = KFold(n_splits=5, shuffle=True)
    #
    clf_f1 = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='f1')
    clf_f1.fit(X_train_SVM, y_encoded_train)
    clf_acc = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='accuracy')
    clf_acc.fit(X_train_SVM, y_encoded_train)
    #
    ######### Record parameters for the best score
    params = clf_f1.best_params_                    # {'C': 100, 'gamma': 0.01}
    f1_gamma_parameter = params.get('gamma')
    f1_C_parameter = params.get('C')
    #
    params = clf_acc.best_params_                   # {'C': 1, 'gamma': 0.1}
    acc_gamma_parameter = params.get('gamma')
    acc_C_parameter = params.get('C')
    #
    text_file.write("f1 model parameter is C: "+str(f1_C_parameter)+", gamma: "+str(f1_gamma_parameter)+"\n")
    text_file.write("acc model parameter is C: "+str(acc_C_parameter)+", gamma: "+str(acc_gamma_parameter)+"\n")
    ######### Loop for each trial
    for c in range(NUM_TRIALS):
        #
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=c)
        #
        ######### F1 Measurement
        svm_f1 = SVC(C=f1_C_parameter,kernel='linear',gamma=f1_gamma_parameter)
        svm_f1.fit(X_train_SVM,y_encoded_train)
        nested_scores_f1 = cross_val_score(svm_f1, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='f1')
        nested_f1[c] = nested_scores_f1.mean()
        text_file.write("Iter " + str(c) + ",F1 nested cross validation score on the cross validation set " + str(nested_f1[c]) + "\n")
        text_file.write(evaluateF1Test(svm_f1,X_test_SVM,y_encoded_test))
        #
        ######### F1 Measurement
        svm_acc = SVC(C=f1_C_parameter,kernel='linear',gamma=f1_gamma_parameter)
        svm_acc.fit(X_train_SVM,y_encoded_train)
        nested_scores_acc = cross_val_score(svm_acc, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='accuracy')
        nested_acc[c] = nested_scores_f1.mean()
        text_file.write("Iter " + str(c) + ",ACC nested cross validation score on the cross validation set " + str(nested_acc[c]) + "\n")
        text_file.write(evaluateACCTest(svm_acc,X_test_SVM,y_encoded_test))
        text_file.write("\n\n")
    #
    text_file.close()

######### RBF Kernel
def SVMKernelRBF(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test):
    ######### Prepare output
    text_file = open("SVMRBF.txt", "w")
    #
    #########  Prepare parameters
    NUM_TRIALS = 5
    p_grid = {"C": [1, 10, 100], "gamma": [0.001,.01, .1]}
    svm = SVC(kernel='rbf')
    #
    ######### Score record
    nested_f1 = np.zeros(NUM_TRIALS)
    nested_acc = np.zeros(NUM_TRIALS)
    #
    ######### Find the best parameters from the non-nested cross validation
    inner_cv = KFold(n_splits=5, shuffle=True)
    #
    clf_f1 = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='f1')
    clf_f1.fit(X_train_SVM, y_encoded_train)
    clf_acc = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='accuracy')
    clf_acc.fit(X_train_SVM, y_encoded_train)
    #
    ######### Record parameters for the best score
    params = clf_f1.best_params_                    # {'C': 100, 'gamma': 0.01}
    f1_gamma_parameter = params.get('gamma')
    f1_C_parameter = params.get('C')
    #
    params = clf_acc.best_params_                   # {'C': 1, 'gamma': 0.1}
    acc_gamma_parameter = params.get('gamma')
    acc_C_parameter = params.get('C')
    #
    text_file.write("f1 model parameter is C: "+str(f1_C_parameter)+", gamma: "+str(f1_gamma_parameter)+"\n")
    text_file.write("acc model parameter is C: "+str(acc_C_parameter)+", gamma: "+str(acc_gamma_parameter)+"\n")
    ######### Loop for each trial
    for c in range(NUM_TRIALS):
        #
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=c)
        #
        ######### F1 Measurement
        svm_f1 = SVC(C=f1_C_parameter,kernel='rbf',gamma=f1_gamma_parameter)
        svm_f1.fit(X_train_SVM,y_encoded_train)
        nested_scores_f1 = cross_val_score(svm_f1, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='f1')
        nested_f1[c] = nested_scores_f1.mean()
        text_file.write("Iter " + str(c) + ",F1 nested cross validation score on the cross validation set " + str(nested_f1[c]) + "\n")
        text_file.write(evaluateF1Test(svm_f1,X_test_SVM,y_encoded_test))
        #
        ######### F1 Measurement
        svm_acc = SVC(C=f1_C_parameter,kernel='rbf',gamma=f1_gamma_parameter)
        svm_acc.fit(X_train_SVM,y_encoded_train)
        nested_scores_acc = cross_val_score(svm_acc, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='accuracy')
        nested_acc[c] = nested_scores_f1.mean()
        text_file.write("Iter " + str(c) + ",ACC nested cross validation score on the cross validation set " + str(nested_acc[c]) + "\n")
        text_file.write(evaluateACCTest(svm_acc,X_test_SVM,y_encoded_test))
        text_file.write("\n\n")
    #
    text_file.close()

def evaluateF1Test(model,X_test_SVM,y_encoded_test):
    #
    result = model.predict(X_test_SVM)
    #
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(result)):
        if((result[i] == 0) and (y_encoded_test[i] == 1)):    # false negative: predicted negative (non disease is 0), actual positive (disease is 1)
            FN += 1
        if((result[i] == 0) and (y_encoded_test[i] == 0)):    # true negative: predicted negative (non disease is 0), actual negative (non disease is 0)
            TN += 1
        if((result[i] == 1) and (y_encoded_test[i] == 1)):    # true positive: predicted positive (disease is 1), actual positive (disease is 1)
            TP += 1
        if((result[i] == 1) and (y_encoded_test[i] == 0)):    # false positive: predicted positive (disease is 1), actual negative (non disease is 0)
            FP += 1
    #
    F1 = (2*TP)/(2*TP+FP+FN)
    #
    return ("On the test set, f1 score is "+str(F1)+"\n" )

def evaluateACCTest(model,X_test_SVM,y_encoded_test):
    #
    result = model.predict(X_test_SVM)
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
    ACC = (TP+TN) / len(y_test)
    #
    return ("On the test set, ACC score is "+str(ACC)+"\n" )
