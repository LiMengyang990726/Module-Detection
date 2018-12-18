import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import xlsxwriter
import subprocess as sp # for clear console: tmp = sp.call('clear',shell=True)

disease_dic = {}
G = nx.Graph
Gc = nx.Graph
diseases = []
componentDic = {}

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
Gc,disease_dic,diseases=readInput()

# average shortest path length:
# rationale: Longer the shortest path distance from the seed node, less relevant
for i in range(len(diseases)):
    print(str(i)+" "+diseases[i])
    if(Gc.has_node(str(diseases[i]))):
        print("Added")
    else:
        diseases.pop(i)

f= open("AvgSP.txt","w+")
f.write("Gene_ID      Average Shortest Path to all Disease genes\n")
for i in Gc.nodes:
    a = 0
    for j in range(len(diseases)):
        a += len(nx.shortest_path(Gc,source=str(i),target=str(diseases[j])))
    f.write(str(a/float(len(diseases)))+"\n")

# local clustering coefficient
# rationale: the higher a node's clustering coefficient in each disease's network,
#            more likely the node belongs to this disease network


# degree centrality

# 
# markov chain centraility (see the proportion of each gene's importance)

# modularity
