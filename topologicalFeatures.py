import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import xlsxwriter
import subprocess as sp # for clear console: tmp = sp.call('clear',shell=True)
import csv
import pandas as pd


# ==============================================================================
#
# Read input
#
# ==============================================================================

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

all_genes_in_network = set(Gc.nodes())
disease_genes = set(diseases) & all_genes_in_network
disease_genes = list(disease_genes)

# ==============================================================================
#
# F1 average shortest path length:
# rationale: Longer the shortest path distance from the seed node, less relevant
#
# ==============================================================================



f = open('AvgSP.csv', mode='w')
fieldnames = ['Gene_ID', 'Average Shortest Path to all Disease genes']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

for i in Gc.nodes:
    a = 0
    for j in range(len(disease_genes)):
        a += len(nx.shortest_path(Gc,source=str(i),target=str(disease_genes[j])))
    f_writer.writerow({'Gene_ID': i, 'Average Shortest Path to all Disease genes': a/float(len(disease_genes))})

f.close()


# ==============================================================================
#
# F2 local clustering coefficient
# rationale: the higher a node's clustering coefficient in each disease's network,
#            more likely the node belongs to this disease network
#
# ==============================================================================

f= open("LocalCC.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'Local Clustering Coefficient']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

neighbors_of_diseases = {}
for i in disease_genes:
    temp = {i: [n for n in Gc.neighbors(i)]}
    neighbors_of_diseases.update(temp)
#
# neighbors = []
# for i in disease_genes:
#     neighbors += [n for n in Gc.neighbors(i)]
#
# uniqueNeighbors = set(neighbors)
# not_connect_to_diseases = all_genes_in_network - uniqueNeighborsneighbors

for key,value in neighbors_of_diseases.items():
    k = Gc.degree(key)
    N = 0
    for j in range(len(value)):
        if (j+1) < len(value):
            for k in range(j+1, len(value)):
                if(Gc.has_edge(value[j],value[k])):
                    N += 1
    denominator = k*(k-1)
    numerator = 2*N
    if (denominator != 0):
        localCC = float(numerator/denominator)
    else:
        localCC = 0
    f_writer.writerow({'Gene_ID': key, 'Local Clustering Coefficient': localCC})


f.close()


# ==============================================================================
#
# F3 degree centrality
# rationale: Higher, more likely to be involved in a more important functional module

# F4 closeness centrality
# rationale: Higher, more functionally important as needs to be communicated quickly.

# F5 betweeness centrality
# rationale: An important node will lie on a higher proportion of the paths.

# F6 eigenvector centrality
# rationale: the influence of a node in a network

# F7 Percolation centrality
# rationale: importance of a node in purely topological terms, despite the network dynamic

# F8 pagerank
# rationale: the notion of how central a node is in a networkx relative to a particular node
#
# ==============================================================================

f = open("Feature34.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'DegreeCentrality','ClosenessCentrality']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

s1 = list(Gc)

s2 = []
dic2 = nx.degree_centrality(Gc)
for key,value in dic2.items():
    s2 += [value]

s3 = []
dic3 = nx.closeness_centrality(Gc)
for key,value in dic3.items():
    s3 += [value]

for i in range(len(s1)):
    f_writer.writerow({'Gene_ID': s1[i], 'DegreeCentrality': s2[i], 'ClosenessCentrality': s3[i]})

f.close()

# ==============================================================================

f = open("Feature5.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'BetweennessCentrality','EigenvectoreCentrality','PageRank']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

s1 = list(Gc)

s4 = []
dic4 = nx.betweenness_centrality(Gc)
for key,value in dic4.items():
    s4 += [value]

for i in range(len(s1)):
    f_writer.writerow({'Gene_ID': s1[i],'BetweennessCentrality': s4[i]})

f.close()

# ==============================================================================

f = open("Feature68.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'EigenvectoreCentrality','PageRank']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

s1 = list(Gc)

s5 = []
dic5 = nx.eigenvector_centrality(Gc)
for key,value in dic5.items():
    s5 += [value]

# s6 = []
# dic6 = nx.percolation_centrality(Gc)
# for key,value in dic6.items():
#     s6 += [value]

s7 = []
dic7 = nx.pagerank(Gc)
for key,value in dic7.items():
    s7 += [value]

for i in range(len(s1)):
    f_writer.writerow({'Gene_ID': s1[i], 'EigenvectoreCentrality': s5[i],'PageRank': s7[i]})

f.close()


# ==============================================================================
#
# connectivity significance
#
# My implementation is according to the definition, while by refering to the DIAMOnD algorithm
# The correct way of computing it should be changed into the follows
#
# ==============================================================================

from scipy.special import comb

f= open("ConnectivitySignificance.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'ConnectivitySignificance']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

alpha = 2
for node in Gc.nodes():
    neighbors = set(Gc.neighbors(node))
    N = len(Gc.nodes())
    k = Gc.degree(node)
    ks = 0
    for neighbor in neighbors:
        if neighbor in disease_genes:
            ks += 1
    ks0 = ks + (alpha-1)*ks
    s = len(disease_genes)
    s0 = s + (alpha-1)*s
    n1 = comb(s0,ks0)                               # numerator 1
    n2 = comb((N-s),(k-ks))                         # numerator 2
    d = comb((N+(alpha-1)*ks),(k+(alpha-1)*ks))     # denominator
    if (d == 0):
        pvalue = -1                                 # set those invalid number to nan, meaning those nodes doesn't have connection with disease node
    else:
        pvalue = (n1*n2)/d
    f_writer.writerow({'Gene_ID': node, 'ConnectivitySignificance': pvalue})

f.close()
# Remaining problem:
# There are results "nan" and "runtime error" problem unsolved


# ==============================================================================
#
# articulation protein
#
# ==============================================================================

import sys

sys.setrecursionlimit(1000000000)

class articulationGraph:
    def __init__(self,networkxGraph):
        self.graph =  networkxGraph
        self.Time = 0
    def largestIndex(self):
        temp = nx.to_dict_of_lists(Gc)
        largest = float("-inf")
        for key,value in temp.items():
            keyInt = int(key)
            if (keyInt > largest):
                largest = keyInt
        return largest
    def APUtil(self,u, visited, ap, parent, low, disc, f_writer):
        children = 0
        visited[int(u)] = True
        self.Time += 1
        # recursive all neighbors of this node
        for v in self.graph.neighbors(str(u)):
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if visited[int(v)] == False:
                parent[int(v)] = int(u)
                children += 1
                self.APUtil(v, visited, ap, parent, low, disc, f_writer)
                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                low[int(u)] = min(low[int(u)], low[int(v)])
                # u is an articulation point in following cases
                # (1) u is root of DFS tree and has two or more chilren.
                if parent[int(u)] == -1 and children > 1:
                    ap[int(u)] = 1
                #(2) If u is not root and low value of one of its child is more
                # than discovery value of u.
                elif parent[int(u)] != -1 and low[int(v)] >= disc[int(u)]:
                    ap[int(u)] = 1
                # mark those non-articulation points
                else:
                    ap[int(u)] = 0
                f_writer.writerow({'Gene_ID': v, 'isArticulationPoint': ap[int(u)]})
                # Update low value of u for parent function calls
            elif int(v) != parent[int(u)]:
                low[int(u)] = min(low[int(u)], disc[int(v)])
    #The function to do DFS traversal. It uses recursive APUtil()
    def AP(self):
        f = open("ArticulationPoints.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
        fieldnames = ['Gene_ID', 'isArticulationPoint']
        f_writer = csv.DictWriter(f, fieldnames=fieldnames)
        # largestIndex = largestIndex()
        visited = [False] * (100653324)
        disc = [float("Inf")] * (100653324)
        low = [float("Inf")] * (100653324)
        parent = [-1] * (100653324)
        ap = [-1] * (100653324)           # To store articulation points, 1 represents True, 0 represents False
                                             # -1 represents those indices that don't have a node corresponding to it
        counter = 0
        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in self.graph.nodes:
            if visited[counter] == False:
                self.APUtil(i, visited, ap, parent, low, disc, f_writer)
            counter += 1

articulation = articulationGraph(Gc)
articulation.AP()
# strange result: there is only one non-articular point

# ==============================================================================
#
# modularity
#
# ==============================================================================

# # Global modularity
# from networkx.algorithms.community.quality import modularity
#
# N = len(Gc.nodes())
# m = sum([d.get('weight', 1) for u, v, d in Gc.edges(data=True)])
# q0 = 1.0 / (2.0*m)
#
# # Map node labels to contiguous integers
# label_for_node = dict((i, v) for i, v in enumerate(Gc.nodes()))
# node_for_label = dict((label_for_node[i], i) for i in range(N))
#
# # Calculate degrees
# k_for_label = Gc.degree(Gc.nodes())
# k = [k_for_label[label_for_node[i]] for i in range(N)]
#
# # Initialize community
# communities = dict((i, frozenset([i])) for i in range(N))
#
# # Initial modularity
# partition = [[label_for_node[x] for x in c] for c in communities.values()]
# print(modularity(Gc, partition))

# Local modularity
f= open("Modularity.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'Modularity']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

B = nx.modularity_matrix(Gc)

a = []                                    # store average modularity of a node to all disease nodes
counter1 = 0                              # specify matrix B row

for i in Gc.nodes():
    temp = 0
    counter2 = 0                          # specify matrix B column
    counter3 = 0
    for j in Gc.nodes():
        if((j != i) and (j in disease_genes)): # exclude itself, and only include the modularity with disease nodes
            temp += B[counter1,counter2]
            counter3 += 1
        counter2 += 1
    counter1 += 1
    temp = float(temp/counter3)
    a += [temp]
    # print(i+"  "+str(temp))
    f_writer.writerow({'Gene_ID': i, 'Modularity': temp})

f.close()

# ==============================================================================
#
# merge all files
#
# ==============================================================================

# with open('AvgSP.csv', 'r+') as f:
#     fieldnames = ['Gene_ID', 'Average Shortest Path to all Disease genes']
#     f_writer = csv.DictWriter(f, fieldnames=fieldnames)
#     content = f.read()
#     f.seek(0, 0)
#     f_writer.writerow({'Gene_ID': 'Gene_ID', 'Average Shortest Path to all Disease genes': 'Average Shortest Path to all Disease genes'})

AvgSP = pd.read_csv("AvgSP.csv",index_col=0, names = ['Gene_ID','Average Shortest Path to all Disease genes'])
LocalCC = pd.read_csv("LocalCC.csv", index_col=0, names = ['Gene_ID','Local Clustering Coefficient'])
Feature_34 = pd.read_csv("Feature34.csv", index_col=0, names = ['Gene_ID', 'DegreeCentrality','ClosenessCentrality'])
Feature_5 = pd.read_csv("Feature5.csv", index_col=0, names = ['Gene_ID', 'BetweennessCentrality','EigenvectoreCentrality','PageRank']) # need to drop
Feature_5.drop(['EigenvectoreCentrality','PageRank'], axis = 1, inplace = True)
Feature_68 = pd.read_csv("Feature68.csv", index_col=0, names = ['Gene_ID','EigenvectoreCentrality','PageRank'])
ConnectivitySignificance = pd.read_csv("ConnectivitySignificance.csv", index_col=0, names = ['Gene_ID','ConnectivitySignificance'])
ArticulationPoints = pd.read_csv("ArticulationPoints.csv", index_col=0, names = ['Gene_ID', 'isArticulationPoint'])
Modularity = pd.read_csv("Modularity.csv", index_col=0, names = ['Gene_ID', 'Modularity'])

topoFeatures = AvgSP.join(LocalCC)
topoFeatures = topoFeatures.join(Feature_34)
topoFeatures = topoFeatures.join(Feature_5)
topoFeatures = topoFeatures.join(Feature_68)
topoFeatures = topoFeatures.join(ConnectivitySignificance)
topoFeatures = topoFeatures.join(ArticulationPoints)
topoFeatures = topoFeatures.join(Modularity)
topoFeatures.to_csv("allTopoFeatures.csv",index='Gene_ID',sep=',')

# ArticulationPoints data is skewed

# Notes:
# 1. Due to the computational power limit and restraint of logging in through ssh,
#    all the above computations are only partially done. Please run the whole script
#    to get all output files.
# 2. Networkx's percolation_centrality package has some fallacy, and I might need
#    some time to fix it.
# 3. Networkx doesn't have function to calculate markov centrality, and I am confused
#    by the difference between markov centrality, random walk betweenness, and pagerank.
#    If neccessary, I will try to implement the calculation of random walk betweennes
# 4. There are some remaining problem in calculating connectivity significance, including
#    getting result of "nan" and runtime error (and I have already excluded the case that
#    denominator is zero)
# Questions:
# 1. Shall we convert all proteins represented in Uniprot and GO term form
#    into geneID representation?
# 2. If we are going to use geneID representation, geneID is generic to an individual
#    while it doesn't have topological properties
#    An inspiration: can figure out those potential disease genes and then observe if an
#    individual's gene bar contains that specific gene
LocalCC_norm,DegreeCen_norm,CloseCen_norm,BetweenCen_norm,EigenCen_norm,PageRank_norm,Modu_norm
