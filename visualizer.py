import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def getInput():
    G = nx.read_edgelist("/home2/lime0400/Module-Detection/data/dataset2/ppi_edgelist.csv")
    disease_genes = []
    with open("/home2/lime0400/Module-Detection/data/dataset2/disgenes_uniprot.csv", 'r') as f:
        for line in f.readlines():
            test = line
            disease_genes.append(test)
    return (G,disease_genes)


def predictedDisease(model1,model2,model3):
    train_1 = pd.read_csv("/home2/lime0400/Module-Detection/ML/DS1_train.csv")
    train_1.drop(['Unnamed: 0'], axis=1, inplace=True)
    test_1 = pd.read_csv("/home2/lime0400/Module-Detection/ML/DS1_test.csv")
    test_1.drop(["Unnamed: 0"], axis=1, inplace=True)
    train_2 = pd.read_csv("/home2/lime0400/Module-Detection/ML/DS2_train.csv")
    train_2.drop(["Unnamed: 0"], axis=1, inplace=True)
    test_2 = pd.read_csv("/home2/lime0400/Module-Detection/ML/DS2_test.csv")
    test_2.drop(["Unnamed: 0"], axis=1, inplace=True)
    train_3 = pd.read_csv("/home2/lime0400/Module-Detection/ML/DS3_train.csv")
    train_3.drop(["Unnamed: 0"], axis=1, inplace=True)
    test_3 = pd.read_csv("/home2/lime0400/Module-Detection/ML/DS3_test.csv")
    test_3.drop(["Unnamed: 0"], axis=1, inplace=True)
    #
    # Only the selected features
    fieldnames = ['EigenvectorCentrality', 'Modularity', 'PageRank',
                  'FrequencyA', 'BP', 'CC', 'MF',
                  'FrequencyC', 'FrequencyD', 'FrequencyE',
                  'FrequencyF', 'FrequencyG', 'FrequencyH',
                  'FrequencyI', 'FrequencyK',
                  'FrequencyL', 'FrequencyM', 'FrequencyN',
                  'FrequencyP', 'FrequencyQ',
                  'FrequencyR', 'FrequencyS', 'FrequencyT',
                  'FrequencyV', 'FrequencyW',
                  'FrequencyY',
                  'Aromaticity', 'SSfractionTurn', 'SSfractionSheet']
    dataset_1 = pd.concat([train_1, test_1])
    dataset_2 = pd.concat([train_2, test_2])
    dataset_3 = pd.concat([train_3, test_3])
    #
    with open(model1, 'rb') as file:
        m1 = pickle.load(file)
    with open(model2, 'rb') as file:
        m2 = pickle.load(file)
    with open(model3, 'rb') as file:
        m3 = pickle.load(file)
    #
    predicted_disease_genes = []
    #
    predicted_1 = m1.predict(dataset_1[fieldnames])
    for i in range(len(predicted_1)):
        if (predicted_1[i] == 1):
            temp = dataset_1.iloc[i, :]
            predicted_disease_genes.append(temp['ProteinID'])
    #
    predicted_2 = m2.predict(dataset_2[fieldnames])
    for i in range(len(predicted_2)):
        if (predicted_2[i] == 1):
            temp = dataset_2.iloc[i, :]
            predicted_disease_genes.append(temp['ProteinID'])
    #
    predicted_3 = m3.predict(dataset_3[fieldnames])
    for i in range(len(predicted_3)):
        if (predicted_3[i] == 1):
            temp = dataset_3.iloc[i, :]
            predicted_disease_genes.append(temp['ProteinID'])
            #
    predicted_disease_genes = list(set(predicted_disease_genes))
    #
    return predicted_disease_genes


def plotTwoDegreeNeighbor(G, disease_genes):
    ## Get all second order neighbors & actual disease genes
    neighbors = []
    for i in disease_genes:
        try:
            first = [n for n in G.neighbors(i[:6])]
            for j in first:
                try:
                    second = [n for n in G.neighbors(j[:6])]
                except:
                    print(j + " is not in the graph")
            neighbors += first
            neighbors += second
        except:
            print(i + " is not in the graph")
    neighbors = set(neighbors)
    subgraph_nodes = list(neighbors)
    #
    ## Get predicted disease node
    model1 = "/home2/lime0400/Module-Detection/ML/random_forest_model_" + str(1) + ".pkl"
    model2 = "/home2/lime0400/Module-Detection/ML/random_forest_model_" + str(2) + ".pkl"
    model3 = "/home2/lime0400/Module-Detection/ML/random_forest_model_" + str(3) + ".pkl"
    predicted_disease_genes = predictedDisease(model1,model2,model3)
    #
    ## Plot subgraph
    print("number of total nodes " + str(len(G.nodes)) + "\n")
    print("number of plotted nodes " + str(len(subgraph_nodes)) + "\n")
    print("number of predicted disease nodes " + str(len(predicted_disease_genes)) + "\n")
    print("number of actual disease nodes " + str(len(disease_genes)) + "\n")
    subG = G.subgraph(subgraph_nodes)
    disG = G.subgraph(disease_genes)
    preDisG = G.subgraph(predicted_disease_genes)
    nx.draw(subG, node_color='black')
    nx.draw(disG, node_color='red')
    nx.draw(preDisG, node_color='blue')
    plt.show()


###########################################
# The main class
###########################################
G, disease_genes = getInput()
plotTwoDegreeNeighbor(G, disease_genes)


