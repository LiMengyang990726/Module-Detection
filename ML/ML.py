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
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=2019)
    return (X_train, X_test, y_train, y_test)

##################################
# Method 1: PCA + K-Means
##################################

######### PCA
def PCA(X_noID):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_noID)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])

    ######### PCA Visualization
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    f1 = principalDf['principal component 1'].values
    f2 = principalDf['principal component 2'].values
    plt.scatter(f1, f2, c='black', s=7)
    plt.show()

    ######### PCA Data split
    XPCA = np.array(list(zip(f1, f2)))
    return XPCA


######### Kmeans Clustering
def Kmeans(X_train_PCA, X_test_PCA, X_train, X_test, y_train, y_test):
    from sklearn.cluster import KMeans

    n_clusters = 3
    kmeans = KMeans(n_clusters,random_state = 2019)
    km = kmeans.fit(X_train_PCA)          # train on the training set only
    labels = kmeans.predict(X_test_PCA)   # should predict on the test set
    centroids = kmeans.cluster_centers_

    # Write output summary
    text_file = open("KmeansOutput.txt", "w")

    #############################################
    # Assume Cluster 0 is disease cluster
    #############################################
    ######### Assume positive is disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] == 0) and (y_test[i] == 1)):    # Label is in cluster 0 (assume the disease cluster), test is 1 (disease cluster)
            TP += 1
        if((labels[i] == 0) and (y_test[i] == 0)):    # Label is in cluster 0 (assume the disease cluster), test is 0 (not in disease cluster)
            FP += 1
        if((labels[i] != 0) and (y_test[i] == 1)):    # Label is not in cluster 0 (assume the disease cluster), test is 1 (disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 0 is considered as target, disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")
    ######### Assume positive is non-disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] != 0) and (y_test[i] == 0)):    # Label is not in cluster 0 (assume the disease cluster), test is 0 (non disease cluster)
            TP += 1
        if((labels[i] != 0) and (y_test[i] == 1)):    # Label is not in cluster 0 (assume the disease cluster), test is 1 (disease cluster)
            FP += 1
        if((labels[i] == 0) and (y_test[i] == 0)):    # Label is in cluster 0 (assume the disease cluster), test is 0 (non disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 0 is considered as target, non-disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")

    #############################################
    # Assume Cluster 0 is non disease cluster
    #############################################
    ######### Assume positive is disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] != 0) and (y_test[i] == 1)):    # Label is not in cluster 0 (disease cluster), test is 1 (disease cluster)
            TP += 1
        if((labels[i] != 0) and (y_test[i] == 0)):    # Label is not in cluster 0 (disease cluster), test is 0 (not in disease cluster)
            FP += 1
        if((labels[i] == 0) and (y_test[i] == 1)):    # Label is in cluster 0 (non disease cluster), test is 1 (disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 0 is considered as non target, disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")
    ######### Assume positive is non-disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] == 0) and (y_test[i] == 0)):    # Label is in cluster 0 (non disease cluster), test is 0 (non disease cluster)
            TP += 1
        if((labels[i] == 0) and (y_test[i] == 1)):    # Label is in cluster 0 (non disease cluster), test is 1 (disease cluster)
            FP += 1
        if((labels[i] != 0) and (y_test[i] == 0)):    # Label is not in cluster 0 (disease cluster), test is 0 (non disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 0 is considered as non target, non-disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")

    #############################################
    # Assume Cluster 1 is disease cluster
    #############################################
    ######### Assume positive is disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] == 1) and (y_test[i] == 1)):    # Label is in cluster 1 (assume the disease cluster), test is 1 (disease cluster)
            TP += 1
        if((labels[i] == 1) and (y_test[i] == 0)):    # Label is in cluster 1 (assume the disease cluster), test is 0 (not in disease cluster)
            FP += 1
        if((labels[i] != 1) and (y_test[i] == 1)):    # Label is not in cluster 1 (assume the disease cluster), test is 1 (disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 1 is considered as target, disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")
    ######### Assume positive is non-disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] != 1) and (y_test[i] == 0)):    # Label is not in cluster 1 (assume the disease cluster), test is 0 (non disease cluster)
            TP += 1
        if((labels[i] != 1) and (y_test[i] == 1)):    # Label is not in cluster 1 (assume the disease cluster), test is 1 (disease cluster)
            FP += 1
        if((labels[i] == 1) and (y_test[i] == 0)):    # Label is in cluster 1 (assume the disease cluster), test is 0 (non disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 1 is considered as target, non-disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")

    #############################################
    # Assume Cluster 1 is non disease cluster
    #############################################
    ######### Assume positive is disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] != 1) and (y_test[i] == 1)):    # Label is not in cluster 1 (disease cluster), test is 1 (disease cluster)
            TP += 1
        if((labels[i] != 1) and (y_test[i] == 0)):    # Label is not in cluster 1 (disease cluster), test is 0 (not in disease cluster)
            FP += 1
        if((labels[i] == 1) and (y_test[i] == 1)):    # Label is in cluster 1 (non disease cluster), test is 1 (disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 1 is considered as non target, disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")
    ######### Assume positive is non-disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] == 1) and (y_test[i] == 0)):    # Label is in cluster 1 (non disease cluster), test is 0 (non disease cluster)
            TP += 1
        if((labels[i] == 1) and (y_test[i] == 1)):    # Label is in cluster 1 (non disease cluster), test is 1 (disease cluster)
            FP += 1
        if((labels[i] != 1) and (y_test[i] == 0)):    # Label is not in cluster 1 (disease cluster), test is 0 (non disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 1 is considered as non target, non-disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")

    #############################################
    # Assume Cluster 2 is disease cluster
    #############################################
    ######### Assume positive is disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] == 2) and (y_test[i] == 1)):    # Label is in cluster 2 (assume the disease cluster), test is 1 (disease cluster)
            TP += 1
        if((labels[i] == 2) and (y_test[i] == 0)):    # Label is in cluster 2 (assume the disease cluster), test is 0 (not in disease cluster)
            FP += 1
        if((labels[i] != 2) and (y_test[i] == 1)):    # Label is not in cluster 2 (assume the disease cluster), test is 1 (disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 2 is considered as target, disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")
    ######### Assume positive is non-disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] != 2) and (y_test[i] == 0)):    # Label is not in cluster 2 (assume the disease cluster), test is 0 (non disease cluster)
            TP += 1
        if((labels[i] != 2) and (y_test[i] == 1)):    # Label is not in cluster 2 (assume the disease cluster), test is 1 (disease cluster)
            FP += 1
        if((labels[i] == 2) and (y_test[i] == 0)):    # Label is in cluster 2 (assume the disease cluster), test is 0 (non disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 2 is considered as target, non-disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")

    #############################################
    # Assume Cluster 2 is non disease cluster
    #############################################
    ######### Assume positive is disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] != 2) and (y_test[i] == 1)):    # Label is not in cluster 2 (disease cluster), test is 1 (disease cluster)
            TP += 1
        if((labels[i] != 2) and (y_test[i] == 0)):    # Label is not in cluster 2 (disease cluster), test is 0 (not in disease cluster)
            FP += 1
        if((labels[i] == 2) and (y_test[i] == 1)):    # Label is in cluster 2 (non disease cluster), test is 1 (disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 2 is considered as non target, disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")
    ######### Assume positive is non-disease
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if((labels[i] == 2) and (y_test[i] == 0)):    # Label is in cluster 2 (non disease cluster), test is 0 (non disease cluster)
            TP += 1
        if((labels[i] == 2) and (y_test[i] == 1)):    # Label is in cluster 2 (non disease cluster), test is 1 (disease cluster)
            FP += 1
        if((labels[i] != 2) and (y_test[i] == 0)):    # Label is not in cluster 2 (disease cluster), test is 0 (non disease cluster)
            FN += 1
    F1 = (2*TP)/(2*TP+FP+FN)
    TN = len(data) - TP - FP - FN
    ACC = (TP+TN) / len(data)
    text_file.write("When cluster 2 is considered as non target, non-disease is considered as positive, f1 score is " + str(F1) + ", accuracy score is " + str(ACC) + "\n")

    text_file.close()

######### Visualize the results (Here we have many redundant codes)
# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
#
#
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
#
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

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
    text_file = open("SVMRBFOutput.txt", "w")
    #
    #########  Prepare parameters
    NUM_TRIALS = 10
    p_grid = {"C": [1, 10, 100], "gamma": [.01, .1]}
    svm = SVC(kernel="rbf")
    #
    # Score record
    non_nested_f1 = np.zeros(NUM_TRIALS)
    nested_f1 = np.zeros(NUM_TRIALS)
    f1_test_dis = np.zeros(NUM_TRIALS)
    f1_test_nondis = np.zeros(NUM_TRIALS)
    #
    non_nested_acc = np.zeros(NUM_TRIALS)
    nested_acc = np.zeros(NUM_TRIALS)
    acc_test_dis = np.zeros(NUM_TRIALS)
    acc_test_nondis = np.zeros(NUM_TRIALS)
    #
    # Parameter
    f1_gamma_parameters = np.zeros(NUM_TRIALS)
    f1_C_parameters = np.zeros(NUM_TRIALS)
    acc_gamma_parameters = np.zeros(NUM_TRIALS)
    acc_C_parameters = np.zeros(NUM_TRIALS)
    #
    ######### Loop for each trial
    for c in range(NUM_TRIALS):
        # Prepare nested and non-nested corss validation
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=c)
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=c)
        #
        ############################
        # F1 measurement for nested and non-nested cross validation (On train set)
        ############################
        clf_f1 = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='f1')
        clf_f1.fit(X_train_SVM, y_encoded_train)
        #
        # Record parameters for the best score
        params = clf_f1.best_params_
        f1_gamma_parameters[c] = params.get('gamma')
        f1_C_parameters[c] = params.get('C')
        clf = SVC(C=f1_C_parameters[c], kernel='rbf',gamma=f1_gamma_parameters[c],random_state=c)
        #
        # Get F1 score on the GridSearchCV
        non_nested_f1[c] = clf_f1.best_score_
        text_file.write("Iter " + str(c) + ",F1 non-nested score on the GridSearchCV " + str(non_nested_f1[c]) + "\n")
        #
        nested_scores_f1 = cross_val_score(clf_f1, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='f1')
        nested_f1[c] = nested_scores_f1.mean()
        text_file.write("Iter " + str(c) + ",F1 nested cross validation score on the GridSearchCV " + str(nested_f1[c]) + "\n")
        #
        #
        predictions = clf.predict(X_test_SVM)
        # Calculate F1 score on the test set (If disease is considered as positive)
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(predictions)):
            if(predictions[i] == 1 and y_encoded_test[i] == 1):
                TP += 1
            if(predictions[i] == 1 and y_encoded_test[i] == 0):
                FP += 1
            if(predictions[i] == 0 and y_encoded_test[i] == 1):
                FN += 1
        f1_test_dis[c] = (2*TP)/(2*TP+FP+FN)
        text_file.write("Iter " + str(c) + ",F1 non-nested score on the test set (disease is positive) " + str(f1_test_dis) + "\n")
        #
        # Calculate F1 score on the test set (If non disease is considered as positive)
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(predictions)):
            if(predictions[i] == 0 and y_encoded_test[i] == 0):
                TP += 1
            if(predictions[i] == 0 and y_encoded_test[i] == 1):
                FP += 1
            if(predictions[i] == 1 and y_encoded_test[i] == 0):
                FN += 1
        f1_test_nondis[c] = (2*TP)/(2*TP+FP+FN)
        text_file.write("Iter " + str(c) + ",F1 non-nested score on the test set (non disease is positive) " + str(f1_test_nondis) + "\n")
        #
        #
        ############################
        # Accuracy measurement for nested and non-nested cross validation
        ############################
        clf_acc = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='accuracy')
        clf_acc.fit(X_train_SVM, y_encoded_train)
        #
        # Record parameters for the best score
        params = clf_acc.best_params_
        acc_gamma_parameters[c] = params.get('gamma')
        acc_C_parameters[c] = params.get('C')
        clf = SVC(C=acc_C_parameters[c],kernel='rbf',gamma=acc_gamma_parameters[c])
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
        predictions = clf_acc.predict(X_test_SVM)
        # Calculate ACC score on the test set (If disease is considered as positive)
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(predictions)):
            if(predictions[i] == 1 and y_encoded_test[i] == 1):
                TP += 1
            if(predictions[i] == 1 and y_encoded_test[i] == 0):
                FP += 1
            if(predictions[i] == 0 and y_encoded_test[i] == 1):
                FN += 1
        TN = len(data) - TP - FP - FN
        acc_test_dis = (TP+TN) / len(data)
        text_file.write("Iter " + str(c) + ",ACC non-nested score on the test set (disease is positive) " + str(acc_test_dis) + "\n")
        #
        # Calculate ACC score on the test set (If non disease is considered as positive)
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(predictions)):
            if(predictions[i] == 0 and y_encoded_test[i] == 0):
                TP += 1
            if(predictions[i] == 0 and y_encoded_test[i] == 1):
                FP += 1
            if(predictions[i] == 1 and y_encoded_test[i] == 0):
                FN += 1
        TN = len(data) - TP - FP - FN
        acc_test_nondis = (TP+TN) / len(data)
        text_file.write("Iter " + str(c) + ",ACC non-nested score on the test set (non disease is positive) " + str(acc_test_nondis) + "\n")
        #
        #
        print("Now is round: " + str(c) + "\n")
        text_file.write("\n")
    #
    text_file.close()
    return (non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis,
            non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis,
            f1_gamma_parameters, f1_C_parameters, acc_gamma_parameters, acc_C_parameters)


# RBF Kernel
def SVMKernelRBF(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test):
    ######### Prepare output
    text_file = open("SVMRBFOutput.txt", "w")
    #
    #########  Prepare parameters
    NUM_TRIALS = 3
    p_grid = {"C": [1, 10, 100], "gamma": [.01, .1]}
    svm = SVC(kernel="rbf")
    #
    # Score record
    non_nested_f1 = np.zeros(NUM_TRIALS)
    nested_f1 = np.zeros(NUM_TRIALS)
    f1_test_dis = np.zeros(NUM_TRIALS)
    f1_test_nondis = np.zeros(NUM_TRIALS)
    #
    non_nested_acc = np.zeros(NUM_TRIALS)
    nested_acc = np.zeros(NUM_TRIALS)
    acc_test_dis = np.zeros(NUM_TRIALS)
    acc_test_nondis = np.zeros(NUM_TRIALS)
    #
    # Parameter
    f1_gamma_parameters = np.zeros(NUM_TRIALS)
    f1_C_parameters = np.zeros(NUM_TRIALS)
    acc_gamma_parameters = np.zeros(NUM_TRIALS)
    acc_C_parameters = np.zeros(NUM_TRIALS)
    #
    ######### Loop for each trial
    for c in range(NUM_TRIALS):
        # Prepare nested and non-nested corss validation
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=c)
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=c)
        #
        ############################
        # F1 measurement for nested and non-nested cross validation (On train set)
        ############################
        clf_f1 = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='f1')
        clf_f1.fit(X_train_SVM, y_encoded_train)
        #
        # Record parameters for the best score
        params = clf_f1.best_params_
        f1_gamma_parameters[c] = params.get('gamma')
        f1_C_parameters[c] = params.get('C')
        #
        # Get F1 score on the GridSearchCV
        non_nested_f1[c] = clf_f1.best_score_
        text_file.write("Iter " + str(c) + ",F1 non-nested score on the GridSearchCV " + str(non_nested_f1[c]) + "\n")
        #
        nested_scores_f1 = cross_val_score(clf_f1, X=X_train_SVM, y=y_encoded_train, cv=outer_cv,scoring='f1')
        nested_f1[c] = nested_scores_f1.mean()
        text_file.write("Iter " + str(c) + ",F1 nested cross validation score on the GridSearchCV " + str(nested_f1[c]) + "\n")
        #
        #
        clf_1 = SVC(C=f1_C_parameters[c], kernel='rbf',gamma=f1_gamma_parameters[c],random_state=c)
        clf_1.fit(X_train_SVM, y_encoded_train)
        predictions = clf_1.predict(X_test_SVM)
        # Calculate F1 score on the test set (If disease is considered as positive)
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(predictions)):
            if(predictions[i] == 1 and y_encoded_test[i] == 1):
                TP += 1
            if(predictions[i] == 1 and y_encoded_test[i] == 0):
                FP += 1
            if(predictions[i] == 0 and y_encoded_test[i] == 1):
                FN += 1
        f1_test_dis[c] = (2*TP)/(2*TP+FP+FN)
        text_file.write("Iter " + str(c) + ",F1 non-nested score on the test set (disease is positive) " + str(f1_test_dis) + "\n")
        #
        # Calculate F1 score on the test set (If non disease is considered as positive)
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(predictions)):
            if(predictions[i] == 0 and y_encoded_test[i] == 0):
                TP += 1
            if(predictions[i] == 0 and y_encoded_test[i] == 1):
                FP += 1
            if(predictions[i] == 1 and y_encoded_test[i] == 0):
                FN += 1
        f1_test_nondis[c] = (2*TP)/(2*TP+FP+FN)
        text_file.write("Iter " + str(c) + ",F1 non-nested score on the test set (non disease is positive) " + str(f1_test_nondis) + "\n")
        #
        #
        ############################
        # Accuracy measurement for nested and non-nested cross validation
        ############################
        clf_acc = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv,scoring='accuracy')
        clf_acc.fit(X_train_SVM, y_encoded_train)
        #
        # Record parameters for the best score
        params = clf_acc.best_params_
        acc_gamma_parameters[c] = params.get('gamma')
        acc_C_parameters[c] = params.get('C')
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
        clf_2 = SVC(C=acc_C_parameters[c],kernel='rbf',gamma=acc_gamma_parameters[c],random_state=c)
        clf_2.fit(X_train_SVM, y_encoded_train)
        predictions = clf_2.predict(X_test_SVM)
        # Calculate ACC score on the test set (If disease is considered as positive)
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(predictions)):
            if(predictions[i] == 1 and y_encoded_test[i] == 1):
                TP += 1
            if(predictions[i] == 1 and y_encoded_test[i] == 0):
                FP += 1
            if(predictions[i] == 0 and y_encoded_test[i] == 1):
                FN += 1
        TN = len(data) - TP - FP - FN
        acc_test_dis = (TP+TN) / len(data)
        text_file.write("Iter " + str(c) + ",ACC non-nested score on the test set (disease is positive) " + str(acc_test_dis) + "\n")
        #
        # Calculate ACC score on the test set (If non disease is considered as positive)
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(predictions)):
            if(predictions[i] == 0 and y_encoded_test[i] == 0):
                TP += 1
            if(predictions[i] == 0 and y_encoded_test[i] == 1):
                FP += 1
            if(predictions[i] == 1 and y_encoded_test[i] == 0):
                FN += 1
        TN = len(data) - TP - FP - FN
        acc_test_nondis = (TP+TN) / len(data)
        text_file.write("Iter " + str(c) + ",ACC non-nested score on the test set (non disease is positive) " + str(acc_test_nondis) + "\n")
        #
        #
        print("Now is round: " + str(c) + "\n")
        text_file.write("\n")
    #
    text_file.close()
    return (non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis,
            non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis,
            f1_gamma_parameters, f1_C_parameters, acc_gamma_parameters, acc_C_parameters)


def SVMVisualization(non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis,
        non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis,
        f1_gamma_parameters, f1_C_parameters, acc_gamma_parameters, acc_C_parameters, name):
    #
    # F1 plot
    plt.figure()
    plt.subplot(211)
    non_nested_f1_line, = plt.plot(non_nested_f1, color='r')
    nested_f1_line, = plt.plot(nested_f1, color='b')
    f1_test_dis_line, = plt.plot(f1_test_dis, color='m')
    f1_test_nondis_line, = plt.plot(f1_test_nondis,color='y')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_f1_line, nested_f1_line,f1_test_dis_line,f1_test_nondis_line],
               ["Non-Nested CV", "Nested CV", "TestSet disease gene as positive", "TestSet nondisease gene as positive"])
    f1_title = "F1 scores for" + name
    plt.title(f1_title,
              x=.5, y=1.1, fontsize="15")
    plt.show()
    #
    # Accuracy plot
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_f1, color='r')
    nested_line, = plt.plot(nested_f1, color='b')
    plt.ylabel("score", fontsize="14")
    plt.legend([non_nested_scores_line, nested_line],
               ["Non-Nested CV", "Nested CV"],
               bbox_to_anchor=(0, .4, .5, 0))
    acc_title = "Accuracy scores for" + name
    plt.title(acc_title,
              x=.5, y=1.1, fontsize="15")
    plt.show()



##################################
# Method 3: Random Forest
##################################
def RandomForest(X_train_RF,X_test_RF):
    ######### Prepare output
    # text_file = open("SVMRBFOutput.txt", "w")
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
    print("The best f1 score for randomizedsearchcv is " + str(rf_random_f1.best_score_))
    print("The best acc score for randomizedsearchcv is " + str(rf_random_acc.best_score_))
    #
    f1_params = rf_random_f1.best_params_
    acc_params = rf_random_acc.best_params_
    #
    ######### Make Predictions
    predictions = rf.predict(X_test_RF)
    #
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
    TN = len(predictions) - TP - FP - FN
    ACC = (TP+TN) / len(predictions)
    F1
    ACC

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
    X_train_PCA = PCA(X_train.drop(['ProteinID'],axis = 1))
    X_test_PCA = PCA(X_test.drop(['ProteinID'],axis = 1))
    Kmeans(X_train_PCA, X_test_PCA, X_train, X_test, y_train.tolist(),y_test.tolist())

    ### SVM
    y_encoded_train, y_encoded_test = encodeY(y_train,y_test)
    X_train_SVM = X_train.drop(['ProteinID'],axis = 1).values
    X_test_SVM = X_test.drop(['ProteinID'],axis = 1).values
    # RBF kernel
    non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis,non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis,f1_gamma_parameters, f1_C_parameters, acc_gamma_parameters, acc_C_parameters = SVMKernelRBF(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test)
    SVMVisualization(non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis, non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis, f1_gamma_parameters, f1_C_parameters, acc_gamma_parameters, acc_C_parameters, 'rbf kernel')
    # Linear kernel
    non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis,non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis,f1_gamma_parameters, f1_C_parameters, acc_gamma_parameters, acc_C_parameters = SVMKernelLinear(X_train_SVM, X_test_SVM, y_encoded_train, y_encoded_test)
    SVMVisualization(non_nested_f1, nested_f1, f1_test_dis, f1_test_nondis, non_nested_acc, nested_acc, acc_test_dis, acc_test_nondis, f1_gamma_parameters, f1_C_parameters, acc_gamma_parameters, acc_C_parameters, 'linear kernel')

    # RandomForest
    X_train_RF = X_train.drop(['ProteinID'],axis = 1)
    X_test_RF = X_test.drop(['ProteinID'],axis = 1)
    f1_params, acc_params = RandomForest(X_train_RF,X_test_RF)

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
