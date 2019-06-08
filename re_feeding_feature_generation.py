import csv
import networkx as nx
import pandas as pd
import os
import TopologicalFeatures.topologicalFeatures as TopoFeatures
import SequenceFeatures.sequenceFeature as SequenceFeatures
import numpy as np


def readInput3(path):
    G = nx.read_edgelist("./data/dataset2/ppi_edgelist.csv")
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
    seq_file_non_dis = os.path.join(inputPath, "endgenes_sequence.txt") # These two files haven't been obtained. Need to get the sequence coversion form Uniprot
    seq_output_dis = os.path.join(outputPath,"Disease_genes_sequence_features.csv")
    seq_output_non_dis = os.path.join(outputPath, "Non_disease_genes_sequence_features.csv")
    #
    computeSequenceFeatures(seq_file_dis, seq_output_dis)
    computeSequenceFeatures(seq_file_non_dis, seq_output_non_dis)

def selectedFunctionalFeatures():
    # Require Ms Rama's file

def dataCombination():
    eigen = pd.read_csv("/Users/limengyang/Workspaces/Re-feeding Using New Dataset/EigenvectorCentrality.csv")
    modu = pd.read_csv("/Users/limengyang/Workspaces/Re-feeding Using New Dataset/Modularity.csv")
    page = pd.read_csv("/Users/limengyang/Workspaces/Re-feeding Using New Dataset/PageRank.csv")

    seq_dis = pd.read_csv("/Users/limengyang/Workspaces/Re-feeding Using New Dataset/Disease_genes_sequence_features.csv")
    seq_non_dis = pd.read_csv("/Users/limengyang/Workspaces/Re-feeding Using New Dataset/Non_disease_genes_sequence_features.csv")

    function = pd.read_csv("")

    frames = [eigen, modu, page, seq_dis, seq_non_dis, function]

    return frames

def dataCleaning():

def computeSequenceFeatures(input_file, output_file):
    # read ids and sequences in a dictionary
    seq_data=np.genfromtxt(input_file, dtype=str, delimiter='\t')

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

    for p in range(len(seq_data)):
        protein_id = seq_data[p][0]
        seq = seq_data[p][1]

        frequency = SequenceFeatures.getAAfreq(seq)
        frequencies = [0] * 26
        for key,value in frequency.items():
            alpha = ord(key) - 65
            frequencies[alpha] = value

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

def main():
    # Get Input
    G, disease_genes, non_disease_genes = readInput3("./data/ASD_data/")

    # Topological Features
    selectedTopoFeatures(G, "./Re-feeding Using New Dataset/",disease_genes)

    # Sequential Features
    inputPath = "/Users/limengyang/Workspaces/Module-Detection/data/ASD_data/"
    outputPath = "/Users/limengyang/Workspaces/Re-feeding Using New Dataset/"
    selectedSequenceFeatures(inputPath,outputPath)

    # Functional Features
