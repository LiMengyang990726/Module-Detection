##################################
# Method 4: Neural Network
##################################
import pickle
from sklearn.neural_network import MLPClassifier

def NeuralNetwork(X_train_DL, y_train):
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    # accuracy: 0.7237376668601276, F1 score: 0.06666666666666667
    mlp.fit(X_train_DL,y_train)
    #
    pkl_filename = "neural_network_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(mlp, file)

def evaluation(X_test_DL,y_encoded_test):
    # Load Model
    pkl_filename = "neural_network_model.pkl"
    with open(pkl_filename, 'rb') as file:
        mlp = pickle.load(file)
    #
    # Make prediction
    result = mlp.predict(X_test_DL)

    # Calculate the score
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(result)):
        if((result[i] == 0) and (y_encoded_test[i] == 1)):    # false negative: predicted negative (non disease is 0), actual positive (disease is 1)
            FN += 1
        if((result[i] == 0) and (y_encoded_test[i] == 0)):    # true negative: predicted negative (non disease is 0), actual negative (non disease is 0)
            TN += 1
        if((result[i] == 1) and (y_encoded_test[i] == 1)):    # true positive: predicted positive (disease is 1), actual positive (disease is 1)
            TP += 1
        if((result[i] == 1) and (y_encoded_test[i] == 0)):    # false positive: predicted positive (disease is 1), actual negative (non disease is 0)
            FP += 1
    #
    ACC = (TP+TN) / len(y_encoded_test)
    F1 = (2*TP)/(2*TP+FP+FN)
    #
    print("Acc "+str(ACC)+"\n"+"F1 "+str(F1)+"\n")
