import dataPreparation
import KMeansPCA
import numpy as np
import pandas as pd
import SVM
import RandomForest
import NeuralNetwork

############## Read Input
train_1,test_1, train_2,test_2, train_3,test_3 = dataPreparation.output()

train_1.to_csv("DS1_train.csv")
test_1.to_csv("DS1_test.csv")
train_2.to_csv("DS2_train.csv")
test_2.to_csv("DS2_test.csv")
train_3.to_csv("DS3_train.csv")
test_3.to_csv("DS3_test.csv")

# train_1 = pd.read_csv("DS1_train.csv")
# train_1.drop(['Unnamed: 0'],axis = 1, inplace = True)
# test_1 = pd.read_csv("DS1_test.csv")
# test_1.drop(["Unnamed: 0"],axis = 1, inplace = True)
# train_2 = pd.read_csv("DS2_train.csv")
# train_2.drop(["Unnamed: 0"],axis = 1, inplace = True)
# test_2 = pd.read_csv("DS2_test.csv")
# test_2.drop(["Unnamed: 0"],axis = 1, inplace = True)
# train_3 = pd.read_csv("DS3_train.csv")
# train_3.drop(["Unnamed: 0"],axis = 1, inplace = True)
# test_3 = pd.read_csv("DS3_test.csv")
# test_3.drop(["Unnamed: 0"],axis = 1, inplace = True)

X_train_1 = train_1.drop(['Target'],axis=1)
y_train_1 = train_1['Target'].tolist()
X_test_1 = test_1.drop(['Target'],axis=1)
y_test_1 = test_1['Target'].tolist()

X_train_2 = train_2.drop(['Target'],axis=1)
y_train_2 = train_2['Target'].tolist()
X_test_2 = test_2.drop(['Target'],axis=1)
y_test_2 = test_2['Target'].tolist()

X_train_3 = train_3.drop(['Target'],axis=1)
y_train_3 = train_3['Target'].tolist()
X_test_3 = test_3.drop(['Target'],axis=1)
y_test_3 = test_3['Target'].tolist()

############## Kmeans with PCA
# Input preparation for KMeans
X_1_PCA = KMeansPCA.performPCA(pd.concat([X_train_1.drop(['ProteinID'],axis=1),X_test_1.drop(['ProteinID'],axis=1)]))
X_2_PCA = KMeansPCA.performPCA(pd.concat([X_train_2.drop(['ProteinID'],axis=1),X_test_2.drop(['ProteinID'],axis=1)]))
X_3_PCA = KMeansPCA.performPCA(pd.concat([X_train_3.drop(['ProteinID'],axis=1),X_test_3.drop(['ProteinID'],axis=1)]))

y_1_PCA = y_train_1 + y_test_1
y_2_PCA = y_train_2 + y_test_2
y_3_PCA = y_train_3 + y_test_3
KMeansPCA.Kmeans(X_1_PCA,y_1_PCA,1)
KMeansPCA.Kmeans(X_2_PCA,y_2_PCA,2)
KMeansPCA.Kmeans(X_3_PCA,y_3_PCA,3)
# BUG: KMeansPCA.visualization(np.append(X_test_PCA,X_train_PCA))

############## Support Vectore Machine
X_train_1_SVM = X_train_1.drop(['ProteinID'],axis = 1).values
X_test_1_SVM = X_test_1.drop(['ProteinID'],axis = 1).values

X_train_2_SVM = X_train_2.drop(['ProteinID'],axis = 1).values
X_test_2_SVM = X_test_2.drop(['ProteinID'],axis = 1).values

X_train_3_SVM = X_train_3.drop(['ProteinID'],axis = 1).values
X_test_3_SVM = X_test_3.drop(['ProteinID'],axis = 1).values

SVM.SVMKernelRBF(X_train_1_SVM, y_train_1,1)
SVM.evaluation(X_test_1_SVM,y_test_1,1)

SVM.SVMKernelRBF(X_train_2_SVM, y_train_2,2)
SVM.evaluation(X_test_2_SVM,y_test_2,2)

SVM.SVMKernelRBF(X_train_3_SVM, y_train_3,3)
SVM.evaluation(X_test_3_SVM,y_test_3,3)
# HAVEN'T DONE VISUALIZATION

############## Random Forest
X_train_1_RF = X_train_1.drop(['ProteinID'],axis = 1)
X_test_1_RF = X_test_1.drop(['ProteinID'],axis = 1)

X_train_2_RF = X_train_2.drop(['ProteinID'],axis = 1)
X_test_2_RF = X_test_2.drop(['ProteinID'],axis = 1)

X_train_3_RF = X_train_3.drop(['ProteinID'],axis = 1)
X_test_3_RF = X_test_3.drop(['ProteinID'],axis = 1)

RandomForest.randomForest(X_train_1_RF,y_train_1,1)
RandomForest.evaluation(X_test_1_RF,y_test_1,1)

RandomForest.randomForest(X_train_2_RF,y_train_2,2)
RandomForest.evaluation(X_test_2_RF,y_test_2,2)

RandomForest.randomForest(X_train_3_RF,y_train_3,3)
RandomForest.evaluation(X_test_3_RF,y_test_3,3)
# HAVEN'T DONE VISUALIZATION

############## Neural Network
X_train_1_NN, X_test_1_NN = X_train_1_RF, X_test_1_RF
X_train_2_NN, X_test_2_NN = X_train_2_RF, X_test_2_RF
X_train_3_NN, X_test_3_NN = X_train_3_RF, X_test_3_RF

NeuralNetwork.neuralNetwork(X_train_1_NN, y_train_1, 1)
NeuralNetwork.evaluation(X_test_1_NN, y_test_1, 1)

NeuralNetwork.neuralNetwork(X_train_2_NN, y_train_2, 2)
NeuralNetwork.evaluation(X_test_2_NN, y_test_2, 2)

NeuralNetwork.neuralNetwork(X_train_3_NN, y_train_3, 3)
NeuralNetwork.evaluation(X_test_3_NN, y_test_3, 3)
