import pickle
import pandas as pd


###############################################
# This one gives very low score
# 1&2:0.38552382167375027
# 1&3:0.009521298336592222
# 2&3:0.022178860165828814
###############################################

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def union(lst1, lst2):
    final_list = lst1 + lst2
    return final_list

def similarity(X_1,X_2,X_3,train_1,train_2,train_3,test_1,test_2,test_3):
    # Load Model
    pkl_filename = "kmeans_model_"+str(1)+".pkl"
    with open(pkl_filename, 'rb') as file:
        kmeans1 = pickle.load(file)
    #
    pkl_filename = "kmeans_model_"+str(2)+".pkl"
    with open(pkl_filename, 'rb') as file:
        kmeans2 = pickle.load(file)
    #
    pkl_filename = "kmeans_model_"+str(3)+".pkl"
    with open(pkl_filename, 'rb') as file:
        kmeans3 = pickle.load(file)
    #
    #
    # Get prediction
    predictions_1 = kmeans1.predict(X_1)
    predictions_2 = kmeans2.predict(X_2)
    predictions_3 = kmeans3.predict(X_3)
    protein_1 = []
    protein_2 = []
    protein_3 = []
    data1 = pd.concat([train_1,test_1])
    data2 = pd.concat([train_2,test_2])
    data3 = pd.concat([train_3,test_3])
    #
    #
    for i in range(len(X_1)):
        if(predictions_1[i] == 0):
            protein_1.append(data1['ProteinID'].iloc[i])
    #
    #
    for i in range(len(X_2)):
        if(predictions_2[i] == 0):
            protein_2.append(data2['ProteinID'].iloc[i])
    #
    #
    for i in range(len(X_3)):
        if(predictions_3[i] == 0):
            protein_3.append(data3['ProteinID'].iloc[i])

    #
    #
    intersection1_2 = intersection(protein_1, protein_2)
    union1_2 = union(protein_1,protein_2)
    intersection1_3 = intersection(protein_1, protein_3)
    union1_3 = union(protein_1,protein_3)
    intersection2_3 = intersection(protein_2, protein_3)
    union2_3 = union(protein_2,protein_3)
    print("DS1 & DS2: "+str(len(intersection1_2)/len(union1_2))) # 0.19558359621451105
    print("DS1 & DS3: "+str(len(intersection1_3)/len(union1_3))) # 0.00505369551484523
    print("DS2 & DS3: "+str(len(intersection2_3)/len(union2_3))) # 0.011494252873563218
