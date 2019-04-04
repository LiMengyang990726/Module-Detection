##################################
# Method 4: Neural Network
##################################
import pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def neuralNetwork(X_train_NN, y_train, number):
    MAX_ITER = 1000
    mlp = MLPClassifier(max_iter = MAX_ITER)

    parameter_space = {
        'hidden_layer_sizes': [(10,10,10),(5,5,5), (5,5), (5,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam','lbfgs'],
        'alpha': [0.000001, 0.00001, 0.0001, 0.001],
        'learning_rate': ['constant','adaptive'],
    }

    clf = GridSearchCV(mlp, parameter_space, cv=3)
    clf.fit(X_train_NN,y_train)
    # #
    # ######### Record the best parameters
    # #
    # params = clf.best_params_
    # #
    # ######### Make Predictions
    # mlp_good = MLPClassifier(max_iter = MAX_ITER,
    #                          hidden_layer_sizes = params['hidden_layer_sizes'],
    #                         activation = params['activation'],
    #                         solver = params['solver'],
    #                         alpha = params['alpha'],
    #                         learning_rate = params['learning_rate'])
    # mlp_good.fit(X_train_NN, y_train)
    # #
    # On the Cross Validaton Set
    result = cross_val_score(clf, X_train_NN, y_train,cv=10)
    output = pd.DataFrame({'CV1':[],'CV2':[],'CV3':[],'CV4':[],
                        'CV5':[],'CV6':[],'CV7':[],
                        'CV8':[],'CV9':[],'CV10':[]})
    output = output.append({
            'CV1':result[0],'CV2':result[1],'CV3':result[2], 'CV4':result[3],
            'CV5':result[4],'CV6':result[5],'CV7':result[6],
            'CV8':result[7],'CV9':result[8],'CV10':result[9]
    }, ignore_index=True)
    output.to_csv("NeuralNetwork_"+str(number)+".csv")
    #
    # Save model
    pkl_filename = "neural_network_model_"+str(number)+".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

def evaluation(X_test_NN,y_test,number):
    # Load Model
    pkl_filename = "neural_network_model_"+str(number)+".pkl"
    with open(pkl_filename, 'rb') as file:
        mlp = pickle.load(file)
    #
    # Make Prediction
    result = mlp.predict(X_test_NN)
    y_test = y_test
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
    output.to_csv("NeuralNetworkOutput_"+str(number)+".csv")

pkl_filename = "neural_network_model_"+str(3)+".pkl"
with open(pkl_filename, 'rb') as file:
    mlp = pickle.load(file)
#
# Make Prediction
result = mlp.predict(X_test_3_NN)
y_test = y_test_3.tolist()
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
cond = "data set "+str(3)
output = output.append({'Cond':cond,
              'TP': TP,'FN': FN,'FP': FP,'TN':TN,
              'Accuracy':result[0],'Precision':result[1],'Recall':result[2],'F1 score':result[3]},
              ignore_index=True)
output.to_csv("NeuralNetworkOutput_"+str(3)+".csv")
