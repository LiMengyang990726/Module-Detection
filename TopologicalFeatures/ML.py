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
# Method 1
##################################

## Load data
topoFeatures_norm = pd.read_csv('topoFeatures_norm')

## Data preprocessing
topoFeatures_norm.drop(['Unnamed: 0'], axis=1,inplace = True)
topo_norm_without_geneid = topoFeatures_norm.drop(['Gene_ID'],axis = 1)

## PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(topo_norm_without_geneid)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

## PCA Visualization
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
f1 = principalDf['principal component 1'].values
f2 = principalDf['principal component 2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

## Kmeans Clustering
kmeans = KMeans(init='k-means++', n_clusters=70, n_init=10)
kmeans.fit(principalDf)
h = .02     # Step size of the mesh. Decrease to increase the quality of the VQ. point in the mesh [x_min, x_max]x[y_min, y_max].
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = principalDf['principal component 1'].min() - 1, principalDf['principal component 1'].max() + 1
y_min, y_max = principalDf['principal component 2'].min() - 1, principalDf['principal component 2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(principalDf['principal component 1'], principalDf['principal component 1'], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

## Make predictions (according to the most number of disease nodes contained in a cluster)
cluster_map = pd.DataFrame()
cluster_map['data_index'] = topoFeatures_norm['Gene_ID']
cluster_map['cluster'] = kmeans.labels_
cluster_min = cluster_map['cluster'].min()
cluster_max = cluster_map['cluster'].max()
# Read input
def readInput():
    disease_file  = open("../gene-disease0.TSV", 'r')
    diseases = []
    disease_dic = {}
    componentDic = {}
    for line in disease_file:
        li = line.strip()
        if not li.startswith("#"):
            li2 = li.split(' ',1)
            disease_key = li2[0]
            print ("the key is: "+disease_key)
            disease_list = [l for l in (li2[1]).split('/')]
            length = len(disease_list)
            for i in range(length):
                diseases.append(disease_list[i])
            # print (disease_list)
            disease_dic.update({disease_key: disease_list})
    return disease_dic,diseases
disease_dic,diseases=readInput()
# Cluster_disease: row - cluster number; colummn - disease type
cluster_disease = [[0 for x in range(70)] for y in range(70)]
for i in range(cluster_min,cluster_max+1):
    current_cluster = cluster_map[cluster_map.cluster == i]
    current_cluster = list(current_cluster['data_index'])
    y_counter = -1
    for j in disease_dic.values():
        y_counter += 1
        for k in range(len(j)):
            node = int(j[k])
            if(node in current_cluster):
                cluster_disease[i][y_counter] += 1
# Count the index that gives the maximum value in a row
predictions = [0 for x in range(70)]
for i in range(70):
    row = cluster_disease[i]
    index, value = max(enumerate(row), key=operator.itemgetter(1))
    predictions[i] = index
    print(str(predictions[i]) + " " + str(value))

##################################
# Method 2
##################################
# Using Support Vector machine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(topoFeatures_norm, test_size=0.1, random_state=2018)

svclassifier = SVC(kernel='linear')
svclassifier.fit(train_df)

y_pred = svclassifier.predict(X_test)
