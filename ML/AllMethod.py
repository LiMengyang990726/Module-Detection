import dataPreparation
import KMeansPCA
import numpy as np
import SVM
import RandomForest
import NeuralNetwork

############## Read Input
train_1,test_1, train_2,test_2, train_3,test_3 = dataPreparation.output()

X_train_1 = train_1.drop(['Target'],axis=1)
y_train_1 = train_1['Target']
X_test_1 = test_1.drop(['Target'],axis=1)
y_test_1 = test_1['Target']

X_train_2 = train_2.drop(['Target'],axis=1)
y_train_2 = train_2['Target']
X_test_2 = test_2.drop(['Target'],axis=1)
y_test_2 = test_2['Target']

X_train_3 = train_3.drop(['Target'],axis=1)
y_train_3 = train_3['Target']
X_test_3 = test_3.drop(['Target'],axis=1)
y_test_3 = test_3['Target']

############## Kmeans with PCA
# Input preparation for KMeans
X_train_1_PCA = KMeansPCA.performPCA(X_train_1.drop(['ProteinID'],axis = 1))
X_test_1_PCA = KMeansPCA.performPCA(X_test_1.drop(['ProteinID'],axis = 1))

X_train_2_PCA = KMeansPCA.performPCA(X_train_2.drop(['ProteinID'],axis = 1))
X_test_2_PCA = KMeansPCA.performPCA(X_test_2.drop(['ProteinID'],axis = 1))

X_train_3_PCA = KMeansPCA.performPCA(X_train_3.drop(['ProteinID'],axis = 1))
X_test_3_PCA = KMeansPCA.performPCA(X_test_3.drop(['ProteinID'],axis = 1))

KMeansPCA.Kmeans(X_train_1_PCA,1)
KMeansPCA.Kmeans(X_train_1_PCA,2)
KMeansPCA.Kmeans(X_train_1_PCA,3)
KMeansPCA.evaluation(X_test_1_PCA, y_test_1.tolist(),1)
KMeansPCA.evaluation(X_test_2_PCA, y_test_2.tolist(),2)
KMeansPCA.evaluation(X_test_3_PCA, y_test_3.tolist(),3)
# BUG: KMeansPCA.visualization(np.append(X_test_PCA,X_train_PCA))

############## Support Vectore Machine
X_train_1_SVM = X_train_1.drop(['ProteinID'],axis = 1).values
X_test_1_SVM = X_test_1.drop(['ProteinID'],axis = 1).values
y_train_1_SVM, y_test_1_SVM = SVM.encodeY(y_train_1,y_test_1)

X_train_2_SVM = X_train_2.drop(['ProteinID'],axis = 1).values
X_test_2_SVM = X_test_2.drop(['ProteinID'],axis = 1).values
y_train_2_SVM, y_test_2_SVM = SVM.encodeY(y_train_2,y_test_2)

X_train_3_SVM = X_train_3.drop(['ProteinID'],axis = 1).values
X_test_3_SVM = X_test_3.drop(['ProteinID'],axis = 1).values
y_train_3_SVM, y_test_3_SVM = SVM.encodeY(y_train_3,y_test_3)

SVM.SVMKernelRBF(X_train_1_SVM, y_train_1_SVM,1)
SVM.evaluation(X_test_1_SVM,y_test_1_SVM,1)

SVM.SVMKernelRBF(X_train_2_SVM, y_train_2_SVM,2)
SVM.evaluation(X_test_2_SVM,y_test_2_SVM,2)

SVM.SVMKernelRBF(X_train_3_SVM, y_train_3_SVM,3)
SVM.evaluation(X_test_3_SVM,y_test_3_SVM,3)
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

############## Deep Learning
# HAVEN'T MODIFIED
X_train_DL = X_train.drop(['ProteinID'],axis = 1)
X_test_DL = X_test.drop(['ProteinID'],axis = 1)
NeuralNetwork.NeuralNetwork(X_train_DL, y_train)
NeuralNetwork.evaluation(X_test_DL, y_test.tolist())
