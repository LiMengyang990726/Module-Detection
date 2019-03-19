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





######### Visualize the results (Here we have many redundant codes)
def KMeansVisualization(principalComponents,centroids):
    #
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    #
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = principalComponents[:, 0].min() - 1, principalComponents[:, 0].max() + 1
    y_min, y_max = principalComponents[:, 1].min() - 1, principalComponents[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    plt.plot(principalComponents[:, 0], principalComponents[:, 1], 'k.', markersize=2)
    #
    # Plot the centroids as a white X
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()




######### Linear Kernel
def SVMKernelLinear(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test):
    ######### Prepare output
    text_file = open("SVMLinearOutput.txt", "w")
    #
    #########  Prepare parameters
    NUM_TRIALS = 5
    p_grid = {"C": [1, 10, 100], "gamma": [.01, .1]}
    svm = SVC(kernel="linear")
    #
    # Score record
    non_nested_f1 = np.zeros(NUM_TRIALS)
    nested_f1 = np.zeros(NUM_TRIALS)
    #
    non_nested_acc = np.zeros(NUM_TRIALS)
    nested_acc = np.zeros(NUM_TRIALS)
    ######### Loop for each trial
    for c in range(NUM_TRIALS):
        # Prepare nested and non-nested corss validation
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=c)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=c)
        #
        ############################
        # F1 measurement for nested and non-nested cross validation (On train set)
        ############################
        clf_f1 = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='f1')
        clf_f1.fit(X_train_SVM, y_encoded_train)
        #
        # Get F1 score on the GridSearchCV
        non_nested_f1[c] = clf_f1.best_score_
        text_file.write("Iter " + str(c) + ",F1 non-nested score on the GridSearchCV " + str(non_nested_f1[c]) + "\n")
        #
        nested_scores_f1 = cross_val_score(clf_f1, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='f1')
        nested_f1[c] = nested_scores_f1.mean()
        text_file.write("Iter " + str(c) + ",F1 nested cross validation score on the GridSearchCV " + str(nested_f1[c]) + "\n")
        #
        ############################
        # Accuracy measurement for nested and non-nested cross validation
        ############################
        clf_acc = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='accuracy')
        clf_acc.fit(X_train_SVM, y_encoded_train)
        #
        # Get ACC score on the GridSearchCV
        non_nested_acc[c] = clf_acc.best_score_
        text_file.write("Iter " + str(c) + ",ACC non-nested score on the GridSearchCV " + str(non_nested_acc[c]) + "\n")
        #
        nested_scores_acc = cross_val_score(clf_f1, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='accuracy')
        nested_acc[c] = nested_scores_acc.mean()
        text_file.write("Iter " + str(c) + ",ACC nested cross validation score on the GridSearchCV " + str(nested_acc[c]) + "\n")
        #
    #
    # Record parameters for the best score
    params = clf_f1.best_params_
    f1_gamma_parameter = params.get('gamma')
    f1_C_parameter = params.get('C')
    #
    # Record parameters for the best score
    params = clf_acc.best_params_
    acc_gamma_parameter = params.get('gamma')
    acc_C_parameter = params.get('C')
    #
    text_file.close()
    #
    return (non_nested_f1, nested_f1, non_nested_acc, nested_acc,
            f1_gamma_parameter, f1_C_parameter, acc_gamma_parameter, acc_C_parameter)


# RBF Kernel
def SVMKernelRBF(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test):
    ######### Prepare output
    text_file = open("SVMRBFOutput.txt", "w")
    #
    #########  Prepare parameters
    NUM_TRIALS = 5
    p_grid = {"C": [1, 10, 100], "gamma": [.01, .1]}
    svm = SVC(kernel="rbf")
    #
    # Score record
    non_nested_f1 = np.zeros(NUM_TRIALS)
    nested_f1 = np.zeros(NUM_TRIALS)
    #
    non_nested_acc = np.zeros(NUM_TRIALS)
    nested_acc = np.zeros(NUM_TRIALS)
    ######### Loop for each trial
    for c in range(NUM_TRIALS):
        # Prepare nested and non-nested corss validation
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=c)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=c)
        #
        ############################
        # F1 measurement for nested and non-nested cross validation (On train set)
        ############################
        clf_f1 = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='f1')
        clf_f1.fit(X_train_SVM, y_encoded_train)
        #
        # Get F1 score on the GridSearchCV
        non_nested_f1[c] = clf_f1.best_score_
        text_file.write("Iter " + str(c) + ",F1 non-nested score on the GridSearchCV " + str(non_nested_f1[c]) + "\n")
        #
        nested_scores_f1 = cross_val_score(clf_f1, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='f1')
        nested_f1[c] = nested_scores_f1.mean()
        text_file.write("Iter " + str(c) + ",F1 nested cross validation score on the GridSearchCV " + str(nested_f1[c]) + "\n")
        #
        ############################
        # Accuracy measurement for nested and non-nested cross validation
        ############################
        clf_acc = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='accuracy')
        clf_acc.fit(X_train_SVM, y_encoded_train)
        #
        # Get ACC score on the GridSearchCV
        non_nested_acc[c] = clf_acc.best_score_
        text_file.write("Iter " + str(c) + ",ACC non-nested score on the GridSearchCV " + str(non_nested_acc[c]) + "\n")
        #
        nested_scores_acc = cross_val_score(clf_f1, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='accuracy')
        nested_acc[c] = nested_scores_acc.mean()
        text_file.write("Iter " + str(c) + ",ACC nested cross validation score on the GridSearchCV " + str(nested_acc[c]) + "\n")
        #
    #
    # Record parameters for the best score
    params = clf_f1.best_params_
    f1_gamma_parameter = params.get('gamma')
    f1_C_parameter = params.get('C')
    #
    # Record parameters for the best score
    params = clf_acc.best_params_
    acc_gamma_parameter = params.get('gamma')
    acc_C_parameter = params.get('C')
    #
    text_file.close()
    #
    return (non_nested_f1, nested_f1, non_nested_acc, nested_acc,
            f1_gamma_parameter, f1_C_parameter, acc_gamma_parameter, acc_C_parameter)

def findLargest(myList):
    maxIndex = 0
    maxValue = float("-inf")
    for i in range(len(myList)):
        if(myList[i] > maxValue):
            maxIndex = i
            maxValue = myList[i]
    return maxIndex, maxValue

def findNonNestedOrNested(maxNonNested, maxNested):
    if(maxNonNested > maxNested):
        return true             # choose non nested cross validation
    else:
        return false            # choose nested cross validation

def findBestParams(non_nested_f1, nested_f1, non_nested_acc, nested_acc,
                    f1_gamma_parameter, f1_C_parameter, acc_gamma_parameter, acc_C_parameter,
                    X_test_SVM, y_encoded_test):
    maxNonNestedF1Index, maxNonNestedF1Value = findLargest(non_nested_f1)
    maxNestedF1Index, maxNestedF1Value = findLargest(nested_f1)
    F1NonNestedOrNested = findNonNestedOrNested(maxNonNestedF1Value, maxNestedF1Value)
    if(F1NonNestedOrNested):
        cv = KFold(n_splits=5, shuffle=True, random_state=maxNonNestedF1Index)
        clf_f1 = SVC(C=f1_C_parameter, kernel='rbf',gamma=f1_gamma_parameter,random_state=maxNonNestedF1Index)
        scores = cross_val_score(clf_f1, X_test_SVM, y_encoded_test, cv=cv, scoring='f1')
        f1_score = scores.mean()
    else:
        # ???
    maxNonNestedACCIndex, maxNonNestedACCValue = findLargest(non_nested_acc)
    maxNestedACCIndex, maxNestedACCValue = findLargest(nested_acc)
    ACCNonNestedOrNested = findNonNestedOrNested(maxNonNestedACCValue, maxNestedACCValue)
    if(ACCNonNestedOrNested):
        # non nested cross validation is better
    else:
        # nested cross validation is better

    return f1_score, acc_score


def SVMVisualization(non_nested_f1, nested_f1, non_nested_acc, nested_acc, name):
    #
    # create x_value
    x_value = []
    for i in range(len(non_nested_f1)):
        x_value.append(i+1)
    #
    # F1 plot
    plt.figure()
    plt.subplot(211)
    non_nested_f1_line, = plt.plot(x_value,non_nested_f1, c='r')
    nested_f1_line, = plt.plot(x_value,nested_f1, c='b')
    plt.ylabel("score", fontsize="14")
    plt.xlabel("random state", fontsize="14")
    plt.legend([non_nested_f1_line, nested_f1_line],
               ["Non-Nested CV", "Nested CV"])
    f1_title = "F1 scores for " + name
    plt.title(f1_title,
              x=.5, y=1.1, fontsize="15")
    plt.show()
    #
    # Accuracy plot
    plt.figure()
    plt.subplot(211)
    non_nested_acc_line, = plt.plot(x_value, non_nested_acc,c='r')
    nested_acc_line, = plt.plot(x_value, nested_acc, c='b')
    plt.ylabel("score", fontsize="14")
    plt.xlabel("random state", fontsize="14")
    plt.legend([non_nested_acc_line, nested_acc_line],
               ["Non-Nested CV", "Nested CV"],
               bbox_to_anchor=(0, .4, .5, 0))
    acc_title = "Accuracy scores for " + name
    plt.title(acc_title,
              x=1.0, y=1.1, fontsize="15")
    plt.show()


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
    TP = 0
    FP = 0
    FN = 0
    actual = y_test.tolist()
    for i in range(len(predictions)):
        if(predictions[i] == actual[i]):
            TP += 1
        if((predictions[i] == 0) and (actual[i] == 1)):
            FP += 1
        if((predictions[i] == 1) and (actual[i] == 0)):
            FN += 1
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
    TP = 0
    FP = 0
    FN = 0
    actual = y_test.tolist()
    for i in range(len(predictions)):
        if(predictions[i] == actual[i]):
            TP += 1
        if((predictions[i] == 0) and (actual[i] == 1)):
            FP += 1
        if((predictions[i] == 1) and (actual[i] == 0)):
            FN += 1
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(predictions)
    text_file.write("Acc score for the test set at the best parameter is " + str(ACC) + "\n")

# >>> f1_params
# {'n_estimators': 100, 'min_samples_leaf': 1, 'min_samples_split': 5, 'bootstrap': False, 'max_features': 'sqrt', 'max_depth': 105}
# >>> acc_params
# {'n_estimators': 800, 'min_samples_leaf': 1, 'min_samples_split': 10, 'bootstrap': False, 'max_features': 'auto', 'max_depth': None}


##################################
# Main to run ALL
##################################

if __name__ == "__main__":
    # path = "/home2/lime0400/Module-Detection/"                        # If in the server
    path = "/Users/limengyang/Workspaces/Module-Detection/"
    data = readInput(path)
    X_train, X_test, y_train, y_test = split(data)

    # PCA
    X_train_PCA, principalComponents_train = PCA(X_train.drop(['ProteinID'],axis = 1))
    X_test_PCA, principalComponents_test = PCA(X_test.drop(['ProteinID'],axis = 1))
    centroids = Kmeans(X_train_PCA, X_test_PCA, X_train, X_test, y_train.tolist(),y_test.tolist())
    KMeansVisualization(principalComponents_train, centroids)

    ### SVM
    y_encoded_train, y_encoded_test = encodeY(y_train,y_test)
    X_train_SVM = X_train.drop(['ProteinID'],axis = 1).values
    X_test_SVM = X_test.drop(['ProteinID'],axis = 1).values

    # RBF kernel
    non_nested_f1, nested_f1, non_nested_acc, nested_acc, f1_gamma_parameter, f1_C_parameter, acc_gamma_parameter, acc_C_parameter = SVMKernelRBF(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test)

    # RBF Kernel f1 result (before the correction)
    # non_nested_f1 = [0.24352122, 0.23772719, 0.23541424, 0.23622278, 0.24413643]
    # nested_f1 = [0.24351791, 0.23772644, 0.23541527, 0.23622192, 0.2441377 ]
    # f1_test_dis = [0.19, 0.19, 0.19, 0.19, 0.19]
    # f1_test_nondis = [0.82947368, 0.82947368, 0.82947368, 0.82947368, 0.82947368]

    # RBF Kernel acc result
    # non_nested_acc = [0.75541796, 0.75386997, 0.75328947, 0.75386997, 0.75367647]
    # nested_acc = [0.70453057, 0.70336404, 0.69736809, 0.70046867, 0.70472193]
    # acc_test_dis = [0.97510012, 0.97510012, 0.97510012, 0.97510012, 0.97510012]
    # acc_test_nondis = [0.97510012, 0.97510012, 0.97510012, 0.97510012, 0.97510012]

    # RBF Kernel result parameters
    # f1_gamma_parameters = [0.01, 0.01, 0.01, 0.01, 0.01]
    # f1_C_parameters = [100., 100., 100., 100., 100.]
    # acc_gamma_parameters = [0.1, 0.1, 0.1, 0.1, 0.1]
    # acc_C_parameters = [1., 1., 1., 1., 1.]

    f1_rbf_score, acc_rbf_score = findBestParams(non_nested_f1, nested_f1, non_nested_acc, nested_acc,
                    f1_gamma_parameter, f1_C_parameter, acc_gamma_parameter, acc_C_parameter,
                    X_test_SVM, y_encoded_test)

    SVMVisualization(non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis, non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis, 'rbf kernel')

    # Linear kernel
    non_nested_f1, nested_f1, non_nested_acc, nested_acc, f1_gamma_parameter, f1_C_parameter, acc_gamma_parameter, acc_C_parameter = SVMKernelLinear(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test)
    f1_linear_score, acc_linear_score = findBestParams(non_nested_f1, nested_f1, non_nested_acc, nested_acc,
                    f1_gamma_parameter, f1_C_parameter, acc_gamma_parameter, acc_C_parameter,
                    X_test_SVM, y_encoded_test)
    SVMVisualization(non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis, non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis, 'linear kernel')

    # RandomForest
    X_train_RF = X_train.drop(['ProteinID'],axis = 1)
    X_test_RF = X_test.drop(['ProteinID'],axis = 1)
    RandomForest(X_train_RF,X_test_RF,y_train,y_test)

##################################
# Main confusions
##################################
# 1. Which evaluation method shall we use? F1 score or accuracy? A new evaluation method read from new paper
# 2. While doing the above evaluation, I calculated each score in two ways:
#    Through grid search/randomized search/nested cross validation find the best score and the best params
#    Through the manual score calculation from the predictions
# 3. For the model internal parameters optimization, how can I get the model with the best parameter?
#     (E.g. The best f1 score that I get from a cv model is different from the one that model.predict)
# (The score from cross validation search and the one manually calculated are different, and don't knnow how to feed the best params in, do we need to create new and train again?)
# 4. The aim of this script is to ? (get the best scores or the best parameters)
