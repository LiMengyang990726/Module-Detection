import csv
import networkx as nx
import pandas as pd
import os
import TopologicalFeatures.topologicalFeatures as TopoFeatures
import SequenceFeatures.sequenceFeature as SequenceFeatures
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import ML.RandomForest as RandomForest

def readInput3(path):
    G = nx.read_edgelist(os.path.join(path,"ppi_edgelist.csv"))
    disease_genes = pd.read_csv(os.path.join(path,"asd_genes.txt"),names=["ProteinID"])
    disease_genes = disease_genes["ProteinID"].tolist()
    non_disease_genes = pd.read_csv(os.path.join(path,"nm_genes.txt"),names=["ProteinID"])
    non_disease_genes = non_disease_genes["ProteinID"].tolist()
    return (G, disease_genes, non_disease_genes)

def selectedTopoFeatures(G, path,disease_genes):
    TopoFeatures.EigenvectorCentrality(G,path)
    TopoFeatures.Modularity(G,path,disease_genes)
    TopoFeatures.PageRank(G,path)

def selectedSequenceFeatures(inputPath,outputPath):
    seq_file_dis = os.path.join(inputPath, "disgenes_sequence.txt")     # These two files haven't been obtained. Need to get the sequence coversion form Uniprot
    seq_file_non_dis = os.path.join(inputPath, "nondisease_genes_sequence.txt") # These two files haven't been obtained. Need to get the sequence coversion form Uniprot
    seq_output_dis = os.path.join(outputPath,"Disease_genes_sequence_features.csv")
    seq_output_non_dis = os.path.join(outputPath, "Non_disease_genes_sequence_features.csv")
    #
    computeSequenceFeatures(seq_file_dis, seq_output_dis)
    computeSequenceFeatures(seq_file_non_dis, seq_output_non_dis)

def selectedFunctionalFeatures(inputPath,outputPath):
    dis_bp = pd.read_table(os.path.join(inputPath, 'asd_bp'), delim_whitespace=True, names=('ProteinID', 'BP'))
    dis_cc = pd.read_table(os.path.join(inputPath, 'asd_cc'), delim_whitespace=True, names=('ProteinID', 'CC'))
    dis_mf = pd.read_table(os.path.join(inputPath, 'asd_mf'), delim_whitespace=True, names=('ProteinID', 'MF'))
    #
    non_dis_bp = pd.read_table(os.path.join(inputPath, 'nm_bp'), delim_whitespace=True, names=('ProteinID', 'BP'))
    non_dis_cc = pd.read_table(os.path.join(inputPath, 'nm_cc'), delim_whitespace=True, names=('ProteinID', 'CC'))
    non_dis_mf = pd.read_table(os.path.join(inputPath, 'nm_mf'), delim_whitespace=True, names=('ProteinID', 'MF'))
    #
    dis_bpcc = pd.merge(dis_bp, dis_cc, on='ProteinID', how='outer')
    dis = pd.merge(dis_bpcc, dis_mf, on='ProteinID', how='outer')
    dis.head()
    #
    non_dis_bpcc = pd.merge(non_dis_bp, non_dis_cc, on='ProteinID', how='outer')
    non_dis = pd.merge(non_dis_bpcc, non_dis_mf, on='ProteinID', how='outer')
    non_dis.head()
    #
    frames = [dis, non_dis]
    allFunctioinalFeatures = pd.concat(frames)
    allFunctioinalFeatures.drop(['Unnamed: 0'],axis=1, inplace=True)
    allFunctioinalFeatures.to_csv(os.path.join(outputPath, "allFunctionalFeatures.csv"))

def dataCombination(outputPath):
    eigen = pd.read_csv(os.path.join(outputPath, "EigenvectorCentrality.csv"), names = ['ProteinID', 'Eigenvector Centrality'])
    modu = pd.read_csv(os.path.join(outputPath,"Modularity.csv"), names = ['ProteinID', 'Modularity'])
    page = pd.read_csv(os.path.join(outputPath,"PageRank.csv"), names = ['ProteinID', 'PageRank'])
    #
    seq_dis = pd.read_csv(os.path.join(outputPath,"Disease_genes_sequence_features.csv"),
                          names = ['ProteinID', 'FrequencyA','FrequencyB',
                    'FrequencyC','FrequencyD','FrequencyE',
                    'FrequencyF','FrequencyG','FrequencyH',
                    'FrequencyI','FrequencyJ','FrequencyK',
                    'FrequencyL','FrequencyM','FrequencyN',
                    'FrequencyO','FrequencyP','FrequencyQ',
                    'FrequencyR','FrequencyS','FrequencyT',
                    'FrequencyU','FrequencyV','FrequencyW',
                    'FrequencyX','FrequencyY','FrequencyZ',
                    'Aromaticity','SSfraction'])
    seq_dis['Target'] = 1
    seq_non_dis = pd.read_csv(os.path.join(outputPath,"Non_disease_genes_sequence_features.csv"),
                              names = ['ProteinID', 'FrequencyA','FrequencyB',
                    'FrequencyC','FrequencyD','FrequencyE',
                    'FrequencyF','FrequencyG','FrequencyH',
                    'FrequencyI','FrequencyJ','FrequencyK',
                    'FrequencyL','FrequencyM','FrequencyN',
                    'FrequencyO','FrequencyP','FrequencyQ',
                    'FrequencyR','FrequencyS','FrequencyT',
                    'FrequencyU','FrequencyV','FrequencyW',
                    'FrequencyX','FrequencyY','FrequencyZ',
                    'Aromaticity','SSfraction'])
    seq_non_dis['Target'] = 0
    #
    function = pd.read_csv(os.path.join(outputPath,"allFunctionalFeatures.csv"))
    #
    t1 = pd.concat([seq_non_dis,seq_dis])
    t2 = pd.merge(t1,function, on='ProteinID')
    t3 = pd.merge(t2,eigen,on='ProteinID')
    t4 = pd.merge(t3,modu,on='ProteinID')
    t5 = pd.merge(t4,page,on='ProteinID')
    t5.to_csv(os.path.join(outputPath, "AllFeatures.csv"))
    return t5

def dataCleaning(outputPath, df):
    #
    # Drop not relevant columns
    df.drop(['FrequencyB', 'FrequencyJ', 'FrequencyO', 'FrequencyU', 'FrequencyX', 'FrequencyZ'], axis=1, inplace=True)
    #
    # Split the SSfraction
    df1 = splitSSfraction(df)
    #
    # Do the scaling
    df2 = normalization(df1)
    df2.to_csv(os.path.join(outputPath,"CleanFeatures.csv"))
    #
    # Prepare Train Set and Test Set
    split = StratifiedShuffleSplit(test_size=0.3, random_state=42)
    for train_index, test_index in split.split(df2, df2['Target']):
        train_set = df2.iloc[train_index]
        test_set = df2.iloc[test_index]
    train_set.reset_index(inplace=True)
    train_set.drop(['index'], axis=1, inplace=True)
    train_set.to_csv(os.path.join(outputPath, "TrainSet.csv"))
    test_set.reset_index(inplace=True)
    test_set.drop(['index'], axis=1, inplace=True)
    test_set.to_csv(os.path.join(outputPath, "TestSet.csv"))
    #
    # Get x and y
    X_train_set = train_set.drop(['Target'],axis=1)
    y_train_set = train_set['Target'].tolist()
    X_test_set = test_set.drop(['Target'],axis=1)
    y_test_set = test_set['Target'].tolist()
    return X_train_set, y_train_set, X_test_set, y_test_set

# TODO: Continue with model feeding
def feedRandomForestModel(X_train, y_train, X_test, y_test):
    #
    # Prepare RF input
    X_train_RF = X_train.drop(['ProteinID'], axis=1)
    X_test_RF = X_test.drop(['ProteinID'], axis=1)
    #
    # Feed into the model
    RandomForest.randomForest(X_train_RF, y_train, 1)
    RandomForest.evaluation(X_test_RF, y_test, 1)


def computeSequenceFeatures(input_file, output_file):
    # read ids and sequences in a dictionary
    seq_data=np.genfromtxt(input_file, dtype=str, delimiter='\t')
    #
    # write output
    f = open(output_file, mode='w')
    fieldnames = ['Protein_ID', 'FrequencyA','FrequencyB',
                    'FrequencyC','FrequencyD','FrequencyE',
                    'FrequencyF','FrequencyG','FrequencyH',
                    'FrequencyI','FrequencyJ','FrequencyK',
                    'FrequencyL','FrequencyM','FrequencyN',
                    'FrequencyO','FrequencyP','FrequencyQ',
                    'FrequencyR','FrequencyS','FrequencyT',
                    'FrequencyU','FrequencyV','FrequencyW',
                    'FrequencyX','FrequencyY','FrequencyZ',
                    'Aromaticity','SSfraction']
    f_writer = csv.DictWriter(f, fieldnames=fieldnames)
    #
    for p in range(len(seq_data)):
        protein_id = seq_data[p][0]
        seq = seq_data[p][1]
        #
        frequency = SequenceFeatures.getAAfreq(seq)
        frequencies = [0] * 26
        if(frequency is not None):
            for key,value in frequency.items():
                alpha = ord(key) - 65
                frequencies[alpha] = value
        #
        aromaticity = SequenceFeatures.getAromaticity(seq)
        ssfraction = SequenceFeatures.getSSfraction(seq)
        f_writer.writerow({'Protein_ID': protein_id,
                            'FrequencyA': frequencies[0],
                            'FrequencyB': frequencies[1],
                            'FrequencyC': frequencies[2],
                            'FrequencyD': frequencies[3],
                            'FrequencyE': frequencies[4],
                            'FrequencyF': frequencies[5],
                            'FrequencyG': frequencies[6],
                            'FrequencyH': frequencies[7],
                            'FrequencyI': frequencies[8],
                            'FrequencyJ': frequencies[9],
                            'FrequencyK': frequencies[10],
                            'FrequencyL': frequencies[11],
                            'FrequencyM': frequencies[12],
                            'FrequencyN': frequencies[13],
                            'FrequencyO': frequencies[14],
                            'FrequencyP': frequencies[15],
                            'FrequencyQ': frequencies[16],
                            'FrequencyR': frequencies[17],
                            'FrequencyS': frequencies[18],
                            'FrequencyT': frequencies[19],
                            'FrequencyU': frequencies[20],
                            'FrequencyV': frequencies[21],
                            'FrequencyW': frequencies[22],
                            'FrequencyX': frequencies[23],
                            'FrequencyY': frequencies[24],
                            'FrequencyZ': frequencies[25],
                            'Aromaticity': aromaticity,
                            'SSfraction': ssfraction})
    f.close()

def splitSSfraction(result):
    # Process SSfraction tuple
    tup_all = result['SSfraction']
    Helix = []
    Turn = []
    Sheet = []
    #
    tup_all = tup_all.tolist()
    #
    for i in range(len(tup_all)):
        tup = list(tup_all[i])
        tup = tup[1:len(tup) - 1]
        #
        counter = 0
        start = [0] * 3
        end = [0] * 3
        #
        helix_each = []
        turn_each = []
        sheet_each = []
        #
        for i in range(len(tup)):
            if (tup[i] == ','):
                start[counter + 1] = i
                end[counter] = (i - 1)
                counter += 1
        #
        helix_each = tup[0:end[0] + 1]
        helix_float = "".join(helix_each)
        helix_float = float(helix_float)
        Helix.append(helix_float)
        #
        turn_each = tup[start[1] + 2:end[1] + 1]
        turn_float = "".join(turn_each)
        turn_float = float(turn_float)
        Turn.append(turn_float)
        #
        sheet_each = tup[start[2] + 2:]
        sheet_float = "".join(turn_each)
        sheet_float = float(sheet_float)
        Sheet.append(sheet_float)
    #
    result['SSfractionHelix'] = Helix
    result['SSfractionTurn'] = Turn
    result['SSfractionSheet'] = Sheet
    result.drop('SSfraction', axis=1, inplace=True)
    return result


def normalization(result):
    # Normalization
    result['FrequencyA'] = preprocessing.scale(result['FrequencyA'])
    result['FrequencyC'] = preprocessing.scale(result['FrequencyC'])
    result['FrequencyD'] = preprocessing.scale(result['FrequencyD'])
    result['FrequencyE'] = preprocessing.scale(result['FrequencyE'])
    result['FrequencyF'] = preprocessing.scale(result['FrequencyF'])
    result['FrequencyG'] = preprocessing.scale(result['FrequencyG'])
    result['FrequencyH'] = preprocessing.scale(result['FrequencyH'])
    result['FrequencyI'] = preprocessing.scale(result['FrequencyI'])
    result['FrequencyK'] = preprocessing.scale(result['FrequencyK'])
    result['FrequencyL'] = preprocessing.scale(result['FrequencyL'])
    result['FrequencyM'] = preprocessing.scale(result['FrequencyM'])
    result['FrequencyN'] = preprocessing.scale(result['FrequencyN'])
    result['FrequencyP'] = preprocessing.scale(result['FrequencyP'])
    result['FrequencyQ'] = preprocessing.scale(result['FrequencyQ'])
    result['FrequencyR'] = preprocessing.scale(result['FrequencyR'])
    result['FrequencyS'] = preprocessing.scale(result['FrequencyS'])
    result['FrequencyT'] = preprocessing.scale(result['FrequencyT'])
    result['FrequencyV'] = preprocessing.scale(result['FrequencyV'])
    result['FrequencyW'] = preprocessing.scale(result['FrequencyW'])
    result['FrequencyY'] = preprocessing.scale(result['FrequencyY'])
    result['Aromaticity'] = preprocessing.scale(result['Aromaticity'])
    result['Eigenvector Centrality'] = preprocessing.scale(result['Eigenvector Centrality'])
    result['Modularity'] = preprocessing.scale(result['Modularity'])
    result['PageRank'] = preprocessing.scale(result['PageRank'])
    result['SSfractionHelix'] = preprocessing.scale(result['SSfractionHelix'])
    result['SSfractionTurn'] = preprocessing.scale(result['SSfractionTurn'])
    result['SSfractionSheet'] = preprocessing.scale(result['SSfractionSheet'])
    result['BP'] = preprocessing.scale(result['BP'])
    result['CC'] = preprocessing.scale(result['CC'])
    result['MF'] = preprocessing.scale(result['MF'])
    #
    return result

def main():
    #
    inputPath = "./data/ASD_data/"
    outputPath = "./Re-feeding Using New Dataset/"

    # Get Input
    G, disease_genes, non_disease_genes = readInput3(inputPath)

    # Topological Features
    selectedTopoFeatures(G, outputPath,disease_genes)

    # Sequential Features
    selectedSequenceFeatures(inputPath,outputPath)

    # Functional Features
    selectedFunctionalFeatures(inputPath, outputPath)

    # Concat all Features
    df = dataCombination(outputPath)
    (X_train_set, y_train_set, X_test_set, y_test_set) = dataCleaning(outputPath, df)

    # Feed into model
    feedRandomForestModel(X_train_set, y_train_set, X_test_set, y_test_set)

