import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import xlsxwriter
import subprocess as sp # for clear console: tmp = sp.call('clear',shell=True)
import csv
import pandas as pd

def readInput():
    G = nx.read_edgelist("./gene-network.tsv")
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    disease_file  = open("gene-disease0.TSV", 'r')
    diseases = []
    disease_dic = {}
    componentDic = {}
    for line in disease_file:
        li = line.strip()
        if not li.startswith("#"):
            li2 = li.split(' ',1)
            disease_key = li2[0]
            print ("the key is: "+disease_key)
            disease_list = [l for l in (li2[1]).split('/')]
            length = len(disease_list)
            for i in range(length):
                diseases.append(disease_list[i])
            print (disease_list)
            disease_dic.update({disease_key: disease_list})
    return Gc, disease_dic,diseases

# main code here
Gc,disease_dic,diseases=readInput()

all_genes_in_network = set(Gc.nodes())
disease_genes = set(diseases) & all_genes_in_network
disease_genes = list(disease_genes)
all_genes_in_network = list(all_genes_in_network)
pos = nx.spring_layout(Gc)

# nodes
nx.draw_networkx_nodes(Gc, pos,
                       nodelist=diseases,
                       node_color='r',
                       node_size=500,
                       alpha=0.8)
nx.draw_networkx_nodes(Gc, pos,
                       nodelist=diseases,
                       node_color='b',
                       node_size=500,
                       alpha=0.8)

# edges
nx.draw_networkx_edges(Gc, pos, width=1.0, alpha=0.5)
