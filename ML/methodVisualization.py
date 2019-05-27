import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

############## Read Input

train_1 = pd.read_csv("DS1_train.csv")
train_1.drop(['Unnamed: 0'], axis=1, inplace=True)
test_1 = pd.read_csv("DS1_test.csv")
test_1.drop(["Unnamed: 0"], axis=1, inplace=True)
train_2 = pd.read_csv("DS2_train.csv")
train_2.drop(["Unnamed: 0"], axis=1, inplace=True)
test_2 = pd.read_csv("DS2_test.csv")
test_2.drop(["Unnamed: 0"], axis=1, inplace=True)
train_3 = pd.read_csv("DS3_train.csv")
train_3.drop(["Unnamed: 0"], axis=1, inplace=True)
test_3 = pd.read_csv("DS3_test.csv")
test_3.drop(["Unnamed: 0"], axis=1, inplace=True)

############## Orginial on the testing dataset

test_1_disease = test_1[test_1["Target"] == 1]
test_1_disease = test_1_disease["ProteinID"].tolist()

test_2_disease = test_2[test_2["Target"] == 1]
test_2_disease = test_2_disease["ProteinID"].tolist()

test_3_disease = test_3[test_3["Target"] == 1]
test_3_disease = test_3_disease["ProteinID"].tolist()

all_test_disease = test_1_disease + test_2_disease + test_3_disease

# test_1_non_disease = test_1[test_1["Target"] == 0]
# test_1_non_disease = test_1_non_disease["ProteinID"].tolist()

# G = nx.read_edgelist("/Users/limengyang/Workspaces/Module-Detection/data/dataset2/ppi_edgelist.csv")
G = nx.read_edgelist("/home2/lime0400/Module-Detection/data/dataset2/ppi_edgelist.csv")
pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,
                       nodelist=all_test_disease,
                       node_color='r',
                       node_size=1,
                   alpha=0.8)

nx.draw_networkx_nodes(G,pos,
                       nodelist=G.nodes-all_test_disease,
                       node_color='b',
                       node_size=1,
                   alpha=0.8)

nx.draw(G,pos)

plt.axis('off')
plt.savefig("visual3.png") # save as png
plt.show()


############## Predicted on the testing dataset
test_1_RF = test_1.drop(['ProteinID',"Target"],axis = 1)
# pkl_filename = "/Users/limengyang/Workspaces/Module-Detection/ML/RandomForest/random_forest_model_1.pkl"
pkl_filename = "/home2/lime0400/Module-Detection/ML/random_forest_model_1.pkl"

with open(pkl_filename, 'rb') as file:
    rf = pickle.load(file)
    result_1 = rf.predict(test_1_RF)


test_2_RF = test_2.drop(['ProteinID',"Target"],axis = 1)
# pkl_filename = "/Users/limengyang/Workspaces/Module-Detection/ML/RandomForest/random_forest_model_1.pkl"
pkl_filename = "/home2/lime0400/Module-Detection/ML/random_forest_model_2.pkl"

with open(pkl_filename, 'rb') as file:
    rf = pickle.load(file)
    result_2 = rf.predict(test_2_RF)


test_3_RF = test_3.drop(['ProteinID',"Target"],axis = 1)
# pkl_filename = "/Users/limengyang/Workspaces/Module-Detection/ML/RandomForest/random_forest_model_1.pkl"
pkl_filename = "/home2/lime0400/Module-Detection/ML/random_forest_model_3.pkl"

with open(pkl_filename, 'rb') as file:
    rf = pickle.load(file)
    result_3 = rf.predict(test_3_RF)


############## Feature relationship and the target
# Avg SP
test_1_disease = test_1[test_1["Target"] == 1]
test_1_non_disease = test_1[test_1["Target"] == 0]
X_disease = test_1_disease["Average Shortest Path to all Disease genes"].tolist()
X_non_disease = test_1_non_disease["Average Shortest Path to all Disease genes"].tolist()
Y_disease = test_1_disease["Target"].tolist()
Y_non_disease = test_1_non_disease["Target"].tolist()

plt.scatter(X_disease, Y_disease, color='r')
plt.scatter(X_non_disease, Y_non_disease, color='g')
plt.xlabel('Average SP')
plt.ylabel('Predicted Value')
plt.show()
plt.savefig("Relationship of AvgSP and TargetValue")

# Modularity
test_1_disease = test_1[test_1["Target"] == 1]
test_1_non_disease = test_1[test_1["Target"] == 0]
X_disease = test_1_disease["Modularity"].tolist()
X_non_disease = test_1_non_disease["Modularity"].tolist()
Y_disease = test_1_disease["Target"].tolist()
Y_non_disease = test_1_non_disease["Target"].tolist()

plt.scatter(X_disease, Y_disease, color='r')
plt.scatter(X_non_disease, Y_non_disease, color='g')
plt.xlabel('Modularity')
plt.ylabel('Predicted Value')
plt.show()
plt.savefig("Relationship of Modularity and TargetValue")

# BP
test_1_disease = test_1[test_1["Target"] == 1]
test_1_non_disease = test_1[test_1["Target"] == 0]
X_disease = test_1_disease["BP"].tolist()
X_non_disease = test_1_non_disease["BP"].tolist()
Y_disease = test_1_disease["Target"].tolist()
Y_non_disease = test_1_non_disease["Target"].tolist()

plt.scatter(X_disease, Y_disease, color='r')
plt.scatter(X_non_disease, Y_non_disease, color='g')
plt.xlabel('BP')
plt.ylabel('Predicted Value')
plt.show()
plt.savefig("Relationship of BP and TargetValue")

# Frequency E
test_1_disease = test_1[test_1["Target"] == 1]
test_1_non_disease = test_1[test_1["Target"] == 0]
X_disease = test_1_disease["FrequencyE"].tolist()
X_non_disease = test_1_non_disease["FrequencyE"].tolist()
Y_disease = test_1_disease["Target"].tolist()
Y_non_disease = test_1_non_disease["Target"].tolist()

plt.scatter(X_disease, Y_disease, color='r')
plt.scatter(X_non_disease, Y_non_disease, color='g')
plt.xlabel('FrequencyE')
plt.ylabel('Predicted Value')
plt.show()
plt.savefig("Relationship of FrequencyE and TargetValue")

# SSfractionTurn
test_1_disease = test_1[test_1["Target"] == 1]
test_1_non_disease = test_1[test_1["Target"] == 0]
X_disease = test_1_disease["SSfractionTurn"].tolist()
X_non_disease = test_1_non_disease["SSfractionTurn"].tolist()
Y_disease = test_1_disease["Target"].tolist()
Y_non_disease = test_1_non_disease["Target"].tolist()

plt.scatter(X_disease, Y_disease, color='r')
plt.scatter(X_non_disease, Y_non_disease, color='g')
plt.xlabel('SSfractionTurn')
plt.ylabel('Predicted Value')
plt.show()
plt.savefig("Relationship of SSfractionTurn and TargetValue")

# SSfractionSheet
test_1_disease = test_1[test_1["Target"] == 1]
test_1_non_disease = test_1[test_1["Target"] == 0]
X_disease = test_1_disease["SSfractionSheet"].tolist()
X_non_disease = test_1_non_disease["SSfractionSheet"].tolist()
Y_disease = test_1_disease["Target"].tolist()
Y_non_disease = test_1_non_disease["Target"].tolist()

plt.scatter(X_disease, Y_disease, color='r')
plt.scatter(X_non_disease, Y_non_disease, color='g')
plt.xlabel('SSfractionSheet')
plt.ylabel('Predicted Value')
plt.show()
plt.savefig("Relationship of SSfractionSheet and TargetValue")


df = pd.DataFrame(test_1, columns=['Average Shortest Path to all Disease genes', 'Modularity', 'FrequencyE', 'SSfractionTurn', 'SSfractionSheet'])
df.plot.box(grid='True')
plt.show()
plt.savefig("Relationship of important features and predicted values.pngRelationship of important features and predicted values.png")