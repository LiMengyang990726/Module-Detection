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
    params = clf_f1.best_params_
    f1_gamma_parameter = params.get('gamma')
    f1_C_parameter = params.get('C')
    #
    params = clf_acc.best_params_
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
def SVMKernelRBF(X_train_SVM, y_encoded_train,number):
    #
    #########  Prepare parameters
    NUM_TRIALS = 5
    #
    C_range = range(16)
    C = np.power(2,C_range).tolist()
    #
    gamma_range = range(16)
    zeros = np.zeros(16)
    gamma = np.power(2,zeros-gamma_range).tolist()
    p_grid = {"C": C, "gamma": gamma}
    svm = SVC(kernel='rbf')
    #
    ######### Score record
    nested_f1 = np.zeros(NUM_TRIALS)
    nested_acc = np.zeros(NUM_TRIALS)
    #
    ######### Find the best parameters from the non-nested cross validation
    inner_cv = KFold(n_splits=5, shuffle=True)
    #
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
    clf.fit(X_train_SVM, y_encoded_train)
    # #
    # ######### Record parameters for the best score
    # params = clf.best_params_                    # {'C': 100, 'gamma': 0.01}
    # gamma_parameter = params.get('gamma')
    # C_parameter = params.get('C')
    # #
    # ######### Train on the best parameters
    # svm = SVC(C=C_parameter,kernel='rbf',gamma=gamma_parameter)
    # svm.fit(X_train_SVM,y_encoded_train)
    #
    # On the Cross Validaton Set
    result = cross_val_score(clf,X_train_SVM, y_encoded_train,cv=10) # cv = 10
    output = pd.DataFrame({'CV1':[],'CV2':[],'CV3':[],'CV4':[],
                        'CV5':[],'CV6':[],'CV7':[],
                        'CV8':[],'CV9':[],'CV10':[],})
    output = output.append({
            'CV1':result[0],'CV2':result[1],'CV3':result[2], 'CV4':result[3],
            'CV5':result[4],'CV6':result[5],'CV7':result[6],
            'CV8':result[7],'CV9':result[8],'CV10':result[9]
    }, ignore_index=True)
    output.to_csv("SVMRBFCV_"+str(number)+".csv")
    #
    # Save model
    pkl_filename = "svm_rbf_model_"+str(number)+".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)


def evaluation(X_test_SVM,y_test,number):
    # Load Model
    pkl_filename = "svm_rbf_model_"+str(number)+".pkl"
    with open(pkl_filename, 'rb') as file:
        svm = pickle.load(file)
    #
    # Make Prediction
    result = svm.predict(X_test_SVM)
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
    output.to_csv("SVMRBFOutput_"+str(number)+".csv")
