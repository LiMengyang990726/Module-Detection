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
    text_file = open("Output.txt", "w")

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
# Main to run ALL
##################################

if __name__ == "__main__":
    # path = "/home2/lime0400/Module-Detection/"
    path = "/Users/limengyang/Workspaces/Module-Detection/"
    data = readInput(path)

    # PCA
    X_train, X_test, y_train, y_test = split(data)
    X_train_PCA = PCA(X_train.drop(['ProteinID'],axis = 1))
    X_test_PCA = PCA(X_test.drop(['ProteinID'],axis = 1))
    Kmeans(X_train_PCA, X_test_PCA, X_train, X_test, y_train.tolist(),y_test.tolist())


##################################
# Method 2: Support Vector machine with nested and non-nested cross validation
##################################

######### Using Support Vector machine
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np

######### Get Input
data = pd.read_csv("/Users/limengyang/Workspaces/Module-Detection/ML/cleanFeatures.csv")
data.drop(['Unnamed: 0'], axis=1,inplace = True)
data.set_index('ProteinID')
X = data.drop(['Target'],axis = 1)
y = data['target']

######### Split train test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=2019)

######### Solve the error of "Continuous 0"
from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
y_encoded_train = lab_enc.fit_transform(y_train)
print(y_encoded_train)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(y_encoded_train))

y_encoded_test = lab_enc.fit_transform(y_test)
print(y_encoded_test)
print(utils.multiclass.type_of_target(y_test))
print(utils.multiclass.type_of_target(y_test.astype('int')))
print(utils.multiclass.type_of_target(y_encoded_test))

# Prepare parameters
NUM_TRIALS = 30

svm = SVC(kernel="rbf") # try linear kernal as well

p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}

non_nested_f1 = np.zeros(NUM_TRIALS)
nested_f1 = np.zeros(NUM_TRIALS)

non_nested_acc = np.zeros(NUM_TRIALS)
nested_acc = np.zeros(NUM_TRIALS)

# Original scores
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

######### Loop for each trial
for i in range(NUM_TRIALS):
    #
    inner_cv = KFold(n_splits=10, shuffle=True, random_state=i) # try 10 fold
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)
    #
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
    clf.fit(X_train, y_train)
    non_nested_scores[i] = clf.best_score_

    #
    ######### calculate the f1 and accuracy score for non_nested
    predicted = clf.predict(X_test)
    y_test_list = y_test.tolist()
    #
    TP = 0
    FP = 0
    FN = 0
    for j in range(len(predicted)):
        if((predicted[j] == 1) and (y_test_list[j] == 1)):
            TP += 1
        if((predicted[j] == 1) and (y_test_list[j] == 0)):
            FP += 1
        if((predicted[j] == 0) and (y_test_list[j] == 1)):
            FN += 1
    #
    TN = len(predicted) - TP - FP - FN
    #
    F1 = (2*TP)/(2*TP+FP+FN)
    non_nested_f1[i] = F1
    ACC = (TP+TN) / len(predicted)
    non_nested_acc[i] = ACC
    #
    # calculate the score for nested (DON'T KNOW WHICH SCORE DOES IT CALCULATING!)
    nested_scores = cross_val_score(clf, X=X_train, y=y_train, cv=outer_cv)
    nested_acc[i] = nested_scores.mean()

######### Visualization of the result
# As the above model is time consuming to run, the result have been saved as
# nested_scores = [0.75187379, 0.74978296, 0.75187125, 0.7511745 , 0.75204764,
#        0.75187258, 0.75065464, 0.75187343, 0.75187161, 0.75204716,
#        0.75100453, 0.75013212, 0.75186967, 0.75152633, 0.75187379,
#        0.75082534, 0.75187306, 0.75047873, 0.75222065, 0.75100271,
#        0.75187125, 0.75204316, 0.74943331, 0.75187367, 0.75187343,
#        0.75065525, 0.75065501, 0.75012945, 0.75134932, 0.75169848]
# non_nested_scores = [0.75187184, 0.75169772, 0.75204597, 0.75204597, 0.75204597,
#        0.75187184, 0.75134947, 0.75187184, 0.75187184, 0.75204597,
#        0.75204597, 0.75187184, 0.75222009, 0.75152359, 0.75204597,
#        0.75187184, 0.75204597, 0.75204597, 0.75187184, 0.75187184,
#        0.75187184, 0.75204597, 0.75152359, 0.75187184, 0.75187184,
#        0.75204597, 0.75187184, 0.75152359, 0.75204597, 0.75204597]
plt.figure()
plt.subplot(211)
non_nested_scores_line, = plt.plot(non_nested_acc, color='r')
nested_line, = plt.plot(nested_acc, color='b')
plt.ylabel("score", fontsize="14")
plt.legend([non_nested_scores_line, nested_line],
           ["Non-Nested CV", "Nested CV"],
           bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested and Nested Cross Validation Accuracy Scores",
          x=.5, y=1.1, fontsize="15")
plt.show()

##################################
# Method 3: Random Forest
##################################
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

######### Get Input
data = pd.read_csv('cleanFeatures.csv')
data.head()
data.drop(['Unnamed: 0'], axis=1,inplace = True)

######### Train test split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
X = data.drop(['Target','ProteinID'],axis=1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=2018)

######### Create model
X_train, y_train = make_classification(n_samples=1000, n_features=37,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
clf.fit(X_train, y_train)

######### Evaluate Result
predictions = clf.predict(X_test)

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

F1 = (2*TP)/(2*TP+FP+FN)                # F1:0.6612339930151339
TN = len(predictions) - TP - FP - FN    # ACC: 0.49391304347826087
ACC = (TP+TN) / len(predictions)
