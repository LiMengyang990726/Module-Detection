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

##################################
# Data preprocessing
##################################

######### Read input files
data = pd.read_csv('../allFeatures.csv')
data.drop(['Unnamed: 0','Unnamed: 0_x','Unnamed: 0_y','FrequencyB','FrequencyJ','FrequencyO','FrequencyU','FrequencyX','FrequencyZ','Hydropathy','Instability'],axis=1,inplace=True)

######### Change the format of SS fraction
tup_all = data['SSfraction']
Helix = []
Turn = []
Sheet = []

for i in range(len(tup_all)):
    tup = list(tup_all[i])
    tup = tup[1:len(tup)-1]

    counter = 0
    start = [0] * 3
    end = [0] *3

    helix_each = []
    turn_each = []
    sheet_each = []

    for i in range(len(tup)):
        if(tup[i] == ','):
            start[counter+1] = i
            end[counter] = (i-1)
            counter += 1

    helix_each = tup[0:end[0]+1]
    helix_float = "".join(helix_each)
    helix_float = float(helix_float)
    Helix.append(helix_float)

    turn_each = tup[start[1]+2:end[1]+1]
    turn_float = "".join(turn_each)
    turn_float = float(turn_float)
    Turn.append(turn_float)

    sheet_each = tup[start[2]+2:]
    sheet_float = "".join(turn_each)
    sheet_float = float(sheet_float)
    Sheet.append(sheet_float)

data['SSfractionHelix'] = Helix
data['SSfractionTurn'] = Turn
data['SSfractionSheet'] = Sheet
data.drop('SSfraction',axis=1,inplace=True)

######### Drop those with null values as inserting zeros will change the network property
data_without_null = data.dropna()
data = data_without_null
data.head()

######### Data normalization
from sklearn import preprocessing
data['FrequencyA'] = preprocessing.scale(data['FrequencyA'])
data['FrequencyC'] = preprocessing.scale(data['FrequencyC'])
data['FrequencyD'] = preprocessing.scale(data['FrequencyD'])
data['FrequencyE'] = preprocessing.scale(data['FrequencyE'])
data['FrequencyF'] = preprocessing.scale(data['FrequencyF'])
data['FrequencyG'] = preprocessing.scale(data['FrequencyG'])
data['FrequencyH'] = preprocessing.scale(data['FrequencyH'])
data['FrequencyI'] = preprocessing.scale(data['FrequencyI'])
data['FrequencyK'] = preprocessing.scale(data['FrequencyK'])
data['FrequencyL'] = preprocessing.scale(data['FrequencyL'])
data['FrequencyM'] = preprocessing.scale(data['FrequencyM'])
data['FrequencyN'] = preprocessing.scale(data['FrequencyN'])
data['FrequencyP'] = preprocessing.scale(data['FrequencyP'])
data['FrequencyQ'] = preprocessing.scale(data['FrequencyQ'])
data['FrequencyR'] = preprocessing.scale(data['FrequencyR'])
data['FrequencyS'] = preprocessing.scale(data['FrequencyS'])
data['FrequencyT'] = preprocessing.scale(data['FrequencyT'])
data['FrequencyV'] = preprocessing.scale(data['FrequencyV'])
data['FrequencyW'] = preprocessing.scale(data['FrequencyW'])
data['FrequencyY'] = preprocessing.scale(data['FrequencyY'])
data['Aromaticity'] = preprocessing.scale(data['Aromaticity'])
data['Isoelectric'] = preprocessing.scale(data['Isoelectric'])
# data['Target'] = preprocessing.scale(data['Target'])
data['Average Shortest Path to all Disease genes'] = preprocessing.scale(data['Average Shortest Path to all Disease genes'])
data['BetweennessCentrality'] = preprocessing.scale(data['BetweennessCentrality'])
data['ClosenessCentrality'] = preprocessing.scale(data['ClosenessCentrality'])
data['DegreeCentrality'] = preprocessing.scale(data['DegreeCentrality'])
data['EigenvectorCentrality'] = preprocessing.scale(data['EigenvectorCentrality'])
data['HarmonicCentrality'] = preprocessing.scale(data['HarmonicCentrality'])
data['Local Clustering Coefficient'] = preprocessing.scale(data['Local Clustering Coefficient'])
data['Modularity'] = preprocessing.scale(data['Modularity'])
data['PageRank'] = preprocessing.scale(data['PageRank'])
data['SSfractionHelix'] = preprocessing.scale(data['SSfractionHelix'])
data['SSfractionTurn'] = preprocessing.scale(data['SSfractionTurn'])
data['SSfractionSheet'] = preprocessing.scale(data['SSfractionSheet'])

######### Combine the functional features
func = pd.read_csv("../allFunctioinalFeatures.csv")
func.drop('Unnamed: 0',axis=1,inplace=True)
result = pd.merge(data,func,on='ProteinID')
result.to_csv("cleanFeatures.csv",index='ProteinID',sep=',')

##################################
# Method 1: PCA + K-Means
##################################

######### Read Input
data = pd.read_csv('cleanFeatures.csv')
data.head()
data.drop(['Unnamed: 0'], axis=1,inplace = True)

######### PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_noID)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

######### PCA Visualization
%matplotlib inline
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
f1 = principalDf['principal component 1'].values
f2 = principalDf['principal component 2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

######### Split train test set
X = data.drop(['Target','ProteinID'],axis=1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=2018)

######### Kmeans Clustering
from sklearn.cluster import KMeans

n_clusters = 3
kmeans = KMeans(n_clusters)
km = kmeans.fit(X_train)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

cluster_map = pd.DataFrame()
cluster_map['data_index'] = data['ProteinID']
cluster_map['cluster'] = km.labels_

cluster0 = cluster_map[cluster_map.cluster == 0]
cluster0_ID = list(cluster0['data_index'])

cluster1 = cluster_map[cluster_map.cluster == 1]
cluster1_ID = list(cluster1['data_index'])

cluster2 = cluster_map[cluster_map.cluster == 2]
cluster2_ID = list(cluster2['data_index'])

######### Calculate the f1 and accuracy score when cluster 0 is considered as target
TP = 0
FP = 0
FN = 0
for index, row in data.iterrows():
    if((row['Target']==1) and (row['ProteinID'] in cluster0_ID)):
        TP += 1
    if((row['Target']==0) and (row['ProteinID'] in cluster0_ID)):
        FP += 1
    if((row['Target']==1) and (row['ProteinID'] not in cluster0_ID)):
        FN += 1
F1 = (2*TP)/(2*TP+FP+FN)
TN = len(data) - TP - FP - FN
ACC = (TP+TN) / len(data)
print("When cluster 0 is considered as target, f1 score is " + str(F1) + ", accuracy score is " + str(ACC))

######### Calculate the f1 and accuracy score when cluster 1 is considered as target
TP = 0
FP = 0
FN = 0
for index, row in data.iterrows():
    if((row['Target']==1) and (row['ProteinID'] in cluster1_ID)):
        TP += 1
    if((row['Target']==0) and (row['ProteinID'] in cluster1_ID)):
        FP += 1
    if((row['Target']==1) and (row['ProteinID'] not in cluster1_ID)):
        FN += 1
F1 = (2*TP)/(2*TP+FP+FN)
TN = len(data) - TP - FP - FN
ACC = (TP+TN) / len(data)
print("When cluster 1 is considered as target, f1 score is " + str(F1) + ", accuracy score is " + str(ACC))

######### Calculate the f1 and accuracy score when cluster 2 is considered as target
TP = 0
FP = 0
FN = 0
for index, row in data.iterrows():
    if((row['Target']==1) and (row['ProteinID'] in cluster2_ID)):
        TP += 1
    if((row['Target']==0) and (row['ProteinID'] in cluster2_ID)):
        FP += 1
    if((row['Target']==1) and (row['ProteinID'] not in cluster2_ID)):
        FN += 1
F1 = (2*TP)/(2*TP+FP+FN)
TN = len(data) - TP - FP - FN
ACC = (TP+TN) / len(data)
print("When cluster 2 is considered as target, f1 score is " + str(F1) + ", accuracy score is " + str(ACC))

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

######### Using Support Vector machine
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np

######### Solve the error of "Continuous 0"
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

svm = SVC(kernel="rbf")

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
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
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
