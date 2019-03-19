import dataPreparation
import KMeansPCA
import numpy as np
import SVM
import RandomForest

############## Read Input
X_train, X_test, y_train, y_test = dataPreparation.output()

############## Kmeans with PCA
X_train_PCA = KMeansPCA.performPCA(X_train.drop(['ProteinID'],axis = 1))
X_test_PCA = KMeansPCA.performPCA(X_test.drop(['ProteinID'],axis = 1))
KMeansPCA.Kmeans(X_train_PCA)
KMeansPCA.evaluation(X_test_PCA, y_test.tolist())
# BUG: KMeansPCA.visualization(np.append(X_test_PCA,X_train_PCA))

############## Support Vectore Machine
X_train_SVM = X_train.drop(['ProteinID'],axis = 1).values
X_test_SVM = X_test.drop(['ProteinID'],axis = 1).values
y_train_SVM, y_test_SVM = SVM.encodeY(y_train,y_test)
SVM.SVMKernelLinear(X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM)
SVM.SVMKernelRBF(X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM)
# HAVEN'T DONE VISUALIZATION

############## Random Forest
X_train_RF = X_train.drop(['ProteinID'],axis = 1)
X_test_RF = X_test.drop(['ProteinID'],axis = 1)
RandomForest(X_train_RF,X_test_RF, y_train, y_test)
# HAVEN'T DONE VISUALIZATION
