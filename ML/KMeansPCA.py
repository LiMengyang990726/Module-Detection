import dataPreparation
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
import pickle
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import ConfusionMatrix

##################################
# Method 1: PCA + K-Means
##################################

######### PCA
def performPCA(X_noID):
    #
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_noID)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    #
    ######### PCA Visualization
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    f1 = principalDf['principal component 1'].values
    f2 = principalDf['principal component 2'].values
    plt.scatter(f1, f2, c='black', s=7)
    plt.show()
    #
    ######### PCA Data split
    XPCA = np.array(list(zip(f1, f2)))
    return XPCA


######### Kmeans Clustering
def Kmeans(X_PCA,y,number):

    n_clusters = 3
    kmeans = KMeans(n_clusters,random_state = 2019)
    kmeans.fit(X_PCA)          # train on the training set only

    ############################################################################
    #
    # Evaluation
    #
    ############################################################################
    labels = kmeans.predict(X_PCA)
    # Prepare output
    output = pd.DataFrame({'Cond':[],
                  'TP': [],'FN': [],'FP': [],'TN':[],
                  'Accuracy':[],'Precision':[],'Recall':[],'F1 score':[]})
    #############################################
    # Assume Cluster 0 is disease cluster
    #############################################
    ######### disease(1) = positive, nondisease(0) = negative
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(labels)):
        if((labels[i] != 0) and (y[i] == 1)):    # false negative: predicted negative (label is not 0), actual positive (disease is 1)
            FN += 1
        if((labels[i] != 0) and (y[i] == 0)):    # true negative: predicted negative (label is not 0), actual negative (non disease is 0)
            TN += 1
        if((labels[i] == 0) and (y[i] == 1)):    # true positive: predicted positive (label is 0), actual positive (disease 1)
            TP += 1
        if((labels[i] == 0) and (y[i] == 0)):    # false positive: predicted positive (label is 0), actual negative (non disease is 0)
            FP += 1

    result = ConfusionMatrix.confusionMatrix(TP,FN,FP,TN)
    output = output.append({'Cond':'cluster 0 is considered as disease',
                  'TP': TP,'FN': FN,'FP': FP,'TN':TN,
                  'Accuracy':result[0],'Precision':result[1],'Recall':result[2],'F1 score':result[3]},
                  ignore_index=True)

    #############################################
    # Assume Cluster 0 is non disease cluster
    #############################################
    ######### disease(1) = positive, nondisease(0) = negative
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(labels)):
        if((labels[i] == 0) and (y[i] == 1)):    # false negative: predicted negative (label is 0), actual positive (disease is 1)
            FN += 1
        if((labels[i] == 0) and (y[i] == 0)):    # true negative: predicted negative (label is 0), actual negative (non disease is 0)
            TN += 1
        if((labels[i] != 0) and (y[i] == 1)):    # true positive: predicted positive (label is not 0), actual positive (disease 1)
            TP += 1
        if((labels[i] != 0) and (y[i] == 0)):    # false positive: predicted positive (label is not 0), actual negative (non disease is 0)
            FP += 1

    result = ConfusionMatrix.confusionMatrix(TP,FN,FP,TN)
    output = output.append({'Cond':'cluster 0 is considered as non disease',
                  'TP': TP,'FN': FN,'FP': FP,'TN':TN,
                  'Accuracy':result[0],'Precision':result[1],'Recall':result[2],'F1 score':result[3]},
                  ignore_index=True)

    #############################################
    # Assume Cluster 1 is disease cluster
    #############################################
    ######### disease(1) = positive, nondisease(0) = negative
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(labels)):
        if((labels[i] != 1) and (y[i] == 1)):    # false negative: predicted negative (label is not 1), actual positive (disease is 1)
            FN += 1
        if((labels[i] != 1) and (y[i] == 0)):    # true negative: predicted negative (label is not 1), actual negative (non disease is 0)
            TN += 1
        if((labels[i] == 1) and (y[i] == 1)):    # true positive: predicted positive (label is 1), actual positive (disease 1)
            TP += 1
        if((labels[i] == 1) and (y[i] == 0)):    # false positive: predicted positive (label is 1), actual negative (non disease is 0)
            FP += 1

    result = ConfusionMatrix.confusionMatrix(TP,FN,FP,TN)
    output = output.append({'Cond':'cluster 1 is considered as disease',
                  'TP': TP,'FN': FN,'FP': FP,'TN':TN,
                  'Accuracy':result[0],'Precision':result[1],'Recall':result[2],'F1 score':result[3]},
                  ignore_index=True)

    #############################################
    # Assume Cluster 1 is non disease cluster
    #############################################
    ######### disease(1) = positive, nondisease(0) = negative
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(labels)):
        if((labels[i] == 1) and (y[i] == 1)):    # false negative: predicted negative (label is 1), actual positive (disease is 1)
            FN += 1
        if((labels[i] == 1) and (y[i] == 0)):    # true negative: predicted negative (label is 1), actual negative (non disease is 0)
            TN += 1
        if((labels[i] != 1) and (y[i] == 1)):    # true positive: predicted positive (label is not 1), actual positive (disease 1)
            TP += 1
        if((labels[i] != 1) and (y[i] == 0)):    # false positive: predicted positive (label is not 1), actual negative (non disease is 0)
            FP += 1

    result = ConfusionMatrix.confusionMatrix(TP,FN,FP,TN)
    output = output.append({'Cond':'cluster 1 is considered as non disease',
                  'TP': TP,'FN': FN,'FP': FP,'TN':TN,
                  'Accuracy':result[0],'Precision':result[1],'Recall':result[2],'F1 score':result[3]},
                  ignore_index=True)

    #############################################
    # Assume Cluster 2 is disease cluster
    #############################################
    ######### disease(1) = positive, nondisease(0) = negative
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(labels)):
        if((labels[i] != 2) and (y[i] == 1)):    # false negative: predicted negative (label is not 2), actual positive (disease is 1)
            FN += 1
        if((labels[i] != 2) and (y[i] == 0)):    # true negative: predicted negative (label is not 2), actual negative (non disease is 0)
            TN += 1
        if((labels[i] == 2) and (y[i] == 1)):    # true positive: predicted positive (label is 2), actual positive (disease 1)
            TP += 1
        if((labels[i] == 2) and (y[i] == 0)):    # false positive: predicted positive (label is 2), actual negative (non disease is 0)
            FP += 1

    result = ConfusionMatrix.confusionMatrix(TP,FN,FP,TN)
    output = output.append({'Cond':'cluster 2 is considered as disease',
                  'TP': TP,'FN': FN,'FP': FP,'TN':TN,
                  'Accuracy':result[0],'Precision':result[1],'Recall':result[2],'F1 score':result[3]},
                  ignore_index=True)

    #############################################
    # Assume Cluster 2 is non disease cluster
    #############################################
    ######### disease(1) = positive, nondisease(0) = negative
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(labels)):
        if((labels[i] == 2) and (y[i] == 1)):    # false negative: predicted negative (label is 2), actual positive (disease is 1)
            FN += 1
        if((labels[i] == 2) and (y[i] == 0)):    # true negative: predicted negative (label is 2), actual negative (non disease is 0)
            TN += 1
        if((labels[i] != 2) and (y[i] == 1)):    # true positive: predicted positive (label is not 2), actual positive (disease 1)
            TP += 1
        if((labels[i] != 2) and (y[i] == 0)):    # false positive: predicted positive (label is not 2), actual negative (non disease is 0)
            FP += 1

    result = ConfusionMatrix.confusionMatrix(TP,FN,FP,TN)
    output = output.append({'Cond':'cluster 2 is considered as non disease',
                  'TP': TP,'FN': FN,'FP': FP,'TN':TN,
                  'Accuracy':result[0],'Precision':result[1],'Recall':result[2],'F1 score':result[3]},
                  ignore_index=True)

    output.to_csv("KmeansOutput_"+str(number)+".csv")
    ############################################################################
    #
    # Save Model
    #
    ############################################################################    pkl_filename = "kmeans_model_"+str(number)+".pkl"
    pkl_filename = "kmeans_model_"+str(number)+".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(kmeans, file)


def visualization(principalComponents):
    pkl_filename = "kmeans_model_"+str(number)+".pkl"
    with open(pkl_filename, 'rb') as file:
        kmeans = pickle.load(file)

    centroids = kmeans.cluster_centers_
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
