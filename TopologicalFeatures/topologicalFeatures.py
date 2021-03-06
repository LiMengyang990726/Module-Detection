import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import subprocess as sp # for clear console: tmp = sp.call('clear',shell=True)
import csv
import pandas as pd
from scipy.special import comb
import os

# ==============================================================================
#
# Read input
#
# ==============================================================================

# For reading dataset1 type: with edgelist, gene-disease
# def readInput1(edgelist,gene-disease):
#     G = nx.read_edgelist(edgelist)
#     Gc = max(nx.connected_component_subgraphs(G), key=len)
#     disease_file  = open(gene-disease, 'r')
#     diseases = []
#     disease_dic = {}
#     componentDic = {}
#     for line in disease_file:
#         li = line.strip()
#         if not li.startswith("#"):
#             li2 = li.split(' ',1)
#             disease_key = li2[0]
#             print ("the key is: "+disease_key)
#             disease_list = [l for l in (li2[1]).split('/')]
#             length = len(disease_list)
#             for i in range(length):
#                 diseases.append(disease_list[i])
#             print (disease_list)
#             disease_dic.update({disease_key: disease_list})
#     return Gc, disease_dic,diseases


# For reading dataset2 type: with edgelist, disgenes_uniprot,endgenes_uniprot,ndnegenes_uniprot
def readInput2(edgelist, disgenes_uniprot,endgenes_uniprot,ndnegenes_uniprot):
    G = nx.read_edgelist("./data/ppi_edgelist.csv")
    disgenes = []
    with open('./data/disgenes_uniprot.csv', 'r') as f:
        for line in f.readlines():
            test = line
            disgenes.append(test)
    for i in range(len(disgenes)):
        temp = disgenes[i]
        print(temp)
        disgenes[i] = temp[:6]
        print(temp[:6])
    endgenes = []
    with open('./data/endgenes_uniprot.csv', 'r') as f:
        for line in f.readlines():
            test = line
            endgenes.append(test)
    for i in range(len(endgenes)):
        temp = endgenes[i]
        print(temp)
        endgenes[i] = temp[:6]
        print(temp[:6])
    ndnegenes = []
    with open('./data/ndnegenes_uniprot.csv', 'r') as f:
        for line in f.readlines():
            test = line
            ndnegenes.append(test)
    for i in range(len(ndnegenes)):
        temp = ndnegenes[i]
        print(temp)
        ndnegenes[i] = temp[:6]
        print(temp[:6])
    return disgenes,endgenes,ndnegenes,G



# ===============================================================
# F1 average shortest path length:
# rationale: Longer the shortest path distance from the seed node, less relevant
#
# ==============================================================================

def AvgSP(Gc,disease_genes,path):
    f = open(os.path.join(path,'AvgSP.csv'), mode='w')
    fieldnames = ['Gene_ID', 'Average Shortest Path to all Disease genes']
    f_writer = csv.DictWriter(f, fieldnames=fieldnames)

    for i in Gc.nodes:
        a = 0
        for j in range(len(disease_genes)):
            try:
                a += len(nx.shortest_path(Gc,source=str(i),target=str(disease_genes[j])))
            except:
                a += 0
        f_writer.writerow({'Gene_ID': i, 'Average Shortest Path to all Disease genes': a/float(len(disease_genes))})

    f.close()



# ==============================================================================
#
# F2 local clustering coefficient
# rationale: the higher a node's clustering coefficient in each disease's network,
#            more likely the node belongs to this disease network
#
# ==============================================================================

def LocalCC(Gc,disease_genes,path):
    f= open(os.path.join(path,"LocalCC.csv"),mode='w') # Only the first level neighbors of disease nodes are shown here
    fieldnames = ['Gene_ID', 'Local Clustering Coefficient']
    f_writer = csv.DictWriter(f, fieldnames=fieldnames)

    neighbors_of_diseases = {}
    for i in disease_genes:
        try:
            temp = {i: [n for n in Gc.neighbors(i)]}
            neighbors_of_diseases.update(temp)
        except:
            f_writer.writerow({'Gene_ID':i, 'Local Clustering Coefficient':0})

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
# F2 degree centrality
# rationale: Higher, more likely to be involved in a more important functional module

# F3 closeness centrality
# rationale: Higher, more functionally important as needs to be communicated quickly.

# F4 betweeness centrality
# rationale: An important node will lie on a higher proportion of the paths.

# F5 eigenvector centrality
# rationale: the influence of a node in a network

# (F6 Percolation centrality (NOT WORKING!)
# rationale: importance of a node in purely topological terms, despite the network dynamic)

# F6 pagerank
# rationale: the notion of how central a node is in a networkx relative to a particular node

# (F7 information centrality (FAILED))

# (F8 second order centrality (METHOD DOES NOT EXIST))

# (F9 in degree centrality (ONLY FOR DIRECTED GRAPH))

# (F10 out degree centrality (ONLY FOR DIRECTED GRAPH))

# (F11 katz centrality (FAILS TO CONVERGE))

# (F12 communicability betweenness centrality (FAILED))

# F13 harmonic centrality
#
# ==============================================================================

def DegreeCentrality(Gc,path):
    s1 = list(Gc)

    s2 = []
    dic2 = nx.degree_centrality(Gc)
    for key,value in dic2.items():
        s2 += [value]

    f = open(os.path.join(path,"DegreeCentrality.csv"),mode = 'w')
    fieldnames = ['Gene_ID','DegreeCentrality']
    f_writer = csv.DictWriter(f,fieldnames=fieldnames)
    for i in range(len(s1)):
        f_writer.writerow({'Gene_ID': s1[i], 'DegreeCentrality': s2[i]})

    f.close()

def ClosenessCentrality(Gc,path):
    s1 = list(Gc)

    s3 = []
    dic3 = nx.closeness_centrality(Gc)
    for key,value in dic3.items():
        s3 += [value]

    f = open(os.path.join(path,"ClosenessCentrality.csv"),mode = 'w')
    fieldnames = ['Gene_ID','ClosenessCentrality']
    f_writer = csv.DictWriter(f,fieldnames=fieldnames)
    for i in range(len(s1)):
        f_writer.writerow({'Gene_ID': s1[i], 'ClosenessCentrality': s3[i]})

    f.close()

def BetweennessCentrality(Gc,path):
    s1 = list(Gc)

    s4 = []
    dic4 = nx.betweenness_centrality(Gc)
    for key,value in dic4.items():
        s4 += [value]

    f = open(os.path.join(path,"BetweennessCentrality.csv"),mode = 'w')
    fieldnames = ['Gene_ID','BetweennessCentrality']
    f_writer = csv.DictWriter(f,fieldnames=fieldnames)
    for i in range(len(s1)):
        f_writer.writerow({'Gene_ID': s1[i], 'BetweennessCentrality': s4[i]})

    f.close()

def EigenvectorCentrality(Gc,path):
    s1 = list(Gc)

    s5 = []
    dic5 = nx.eigenvector_centrality(Gc)
    for key,value in dic5.items():
        s5 += [value]

    f = open(os.path.join(path,"EigenvectorCentrality.csv"),mode = 'w')
    fieldnames = ['Gene_ID','EigenvectorCentrality']
    f_writer = csv.DictWriter(f,fieldnames=fieldnames)
    for i in range(len(s1)):
        f_writer.writerow({'Gene_ID': s1[i], 'EigenvectorCentrality': s5[i]})

    f.close()

def PageRank(Gc,path):
    s1 = list(Gc)

    s6 = []
    dic6 = nx.pagerank(Gc)
    for key,value in dic6.items():
        s6 += [value]

    f = open(os.path.join(path,"PageRank.csv"),mode = 'w')
    fieldnames = ['Gene_ID','PageRank']
    f_writer = csv.DictWriter(f,fieldnames=fieldnames)
    for i in range(len(s1)):
        f_writer.writerow({'Gene_ID': s1[i], 'PageRank': s6[i]})

    f.close()

def HarmonicCentrality(Gc,path):
    s1 = list(Gc)

    s13 = []
    dic13 = nx.harmonic_centrality(Gc)
    for key,value in dic13.items():
        s13 += [value]

    f = open(os.path.join(path,"HarmonicCentrality.csv"),mode = 'w')
    fieldnames = ['Gene_ID','HarmonicCentrality']
    f_writer = csv.DictWriter(f,fieldnames=fieldnames)
    for i in range(len(s1)):
        f_writer.writerow({'Gene_ID': s1[i], 'HarmonicCentrality': s13[i]})

    f.close()


# ==============================================================================
#
# connectivity significance
#
# My implementation is according to the definition, while by refering to the DIAMOnD algorithm
# The correct way of computing it should be changed into the follows
#
# Remaining problem:
# There are results "nan" and "runtime error" problem unsolved
#
# ==============================================================================


def ConnectivitySignificance(Gc,disease_genes,path):

    f= open(os.path.join(path,"ConnectivitySignificance.csv"),mode='w') # Only the first level neighbors of disease nodes are shown here
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



# ==============================================================================
#
# articulation protein
# Discarded: as most of the results are articulation point
#
# ==============================================================================

# import sys
#
# sys.setrecursionlimit(1000000000)
#
# class articulationGraph:
#     def __init__(self,networkxGraph):
#         self.graph =  networkxGraph
#         self.Time = 0
#     def largestIndex(self):
#         temp = nx.to_dict_of_lists(Gc)
#         largest = float("-inf")
#         for key,value in temp.items():
#             keyInt = int(key)
#             if (keyInt > largest):
#                 largest = keyInt
#         return largest
#     def APUtil(self,u, visited, ap, parent, low, disc, f_writer):
#         children = 0
#         visited[int(u)] = True
#         self.Time += 1
#         # recursive all neighbors of this node
#         for v in self.graph.neighbors(str(u)):
#             # If v is not visited yet, then make it a child of u
#             # in DFS tree and recur for it
#             if visited[int(v)] == False:
#                 parent[int(v)] = int(u)
#                 children += 1
#                 self.APUtil(v, visited, ap, parent, low, disc, f_writer)
#                 # Check if the subtree rooted with v has a connection to
#                 # one of the ancestors of u
#                 low[int(u)] = min(low[int(u)], low[int(v)])
#                 # u is an articulation point in following cases
#                 # (1) u is root of DFS tree and has two or more chilren.
#                 if parent[int(u)] == -1 and children > 1:
#                     ap[int(u)] = 1
#                 #(2) If u is not root and low value of one of its child is more
#                 # than discovery value of u.
#                 elif parent[int(u)] != -1 and low[int(v)] >= disc[int(u)]:
#                     ap[int(u)] = 1
#                 # mark those non-articulation points
#                 else:
#                     ap[int(u)] = 0
#                 f_writer.writerow({'Gene_ID': v, 'isArticulationPoint': ap[int(u)]})
#                 # Update low value of u for parent function calls
#             elif int(v) != parent[int(u)]:
#                 low[int(u)] = min(low[int(u)], disc[int(v)])
#     #The function to do DFS traversal. It uses recursive APUtil()
#     def AP(self):
#         f = open("ArticulationPoints.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
#         fieldnames = ['Gene_ID', 'isArticulationPoint']
#         f_writer = csv.DictWriter(f, fieldnames=fieldnames)
#         # largestIndex = largestIndex()
#         visited = [False] * (100653324)
#         disc = [float("Inf")] * (100653324)
#         low = [float("Inf")] * (100653324)
#         parent = [-1] * (100653324)
#         ap = [-1] * (100653324)           # To store articulation points, 1 represents True, 0 represents False
#                                              # -1 represents those indices that don't have a node corresponding to it
#         counter = 0
#         # Call the recursive helper function
#         # to find articulation points
#         # in DFS tree rooted with vertex 'i'
#         for i in self.graph.nodes:
#             if visited[counter] == False:
#                 self.APUtil(i, visited, ap, parent, low, disc, f_writer)
#             counter += 1
#
# articulation = articulationGraph(Gc)
# articulation.AP()
# strange result: there is only one non-articular point

# ==============================================================================
#
# modularity
#
# ==============================================================================
def Modularity(Gc,path,disease_genes):

    f = open(os.path.join(path,"Modularity.csv"),mode='w') # Only the first level neighbors of disease nodes are shown here
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
if __name__ == "__main__":

    # All you need to modify is this path and the input path
    path = "/Users/limengyang/Workspaces/Module-Detection/TopologicalFeatures/DataSet2Features/"
    # Prepare the output directory
    if not os.path.exists(path):
        os.makedirs(path)

    # Input path: NEED TO BE ABSOLUTE TYPE
    inputPath = "/Users/limengyang/Workspaces/Module-Detection/data/dataset2/"
    # Dataset1 Type:
    # Gc,disease_dic,diseases = readInput1("/Users/limengyang/Workspaces/Module-Detection/data/dataset1/gene-network.tsv",
    #                                     "/Users/limengyang/Workspaces/Module-Detection/data/dataset1/gene-disease0")
    # all_genes_in_network = set(Gc.nodes())
    # disease_genes = set(diseases) & all_genes_in_network
    # disease_genes = list(disease_genes)

    # Dataset2 Type:
    disease_genes,endgenes,ndnegenes,Gc = readInput2(os.path.join(path,"ppi_edgelist.csv"),
                                                    os.path.join(path,"disgenes_uniprot.csv"),
                                                    os.path.join(path,"endgenes_uniprot.csv"),
                                                    os.path.join(path,"ndnegenes_uniprot.csv"))

    # Get topological features
    AvgSP(Gc,disease_genes,path)
    LocalCC(Gc,disease_genes,path)
    DegreeCentrality(Gc,path)
    ClosenessCentrality(Gc,path)
    BetweennessCentrality(Gc,path)
    EigenvectorCentrality(Gc,path)
    PageRank(Gc,path)
    HarmonicCentrality(Gc,path)
    Modularity(Gc,path,disease_genes)
    # ConnectivitySignificance(Gc,disease_genes,path)

    # Write down all ProteinID
    all_genes = disease_genes + endgenes + ndnegenes
    f= open(os.path.join(path,"ProteinID.csv"),mode='w')
    fieldnames = ['ProteinID']
    f_writer = csv.DictWriter(f, fieldnames=fieldnames)
    for i in range(len(all_genes)):
        f_writer.writerow({'ProteinID':all_genes[i]})
    f.close()

    a = pd.read_csv(os.path.join(path,"AvgSP.csv"),names = ['ProteinID','Average Shortest Path to all Disease genes'])
    b = pd.read_csv(os.path.join(path,"BetweennessCentrality.csv"), names = ['ProteinID','BetweennessCentrality'])
    c = pd.read_csv(os.path.join(path,"ClosenessCentrality.csv"),names = ['ProteinID','ClosenessCentrality'])
    d = pd.read_csv(os.path.join(path,"DegreeCentrality.csv"),names = ['ProteinID','DegreeCentrality'])
    e = pd.read_csv(os.path.join(path,"EigenvectorCentrality.csv"),names = ['ProteinID','EigenvectorCentrality'])
    h = pd.read_csv(os.path.join(path,"HarmonicCentrality.csv"),names = ['ProteinID','HarmonicCentrality'])
    l = pd.read_csv(os.path.join(path,"LocalCC.csv"),names = ['ProteinID','Local Clustering Coefficient'])
    m = pd.read_csv(os.path.join(path,"Modularity.csv"),names = ['ProteinID','Modularity'])
    pa = pd.read_csv(os.path.join(path,"PageRank.csv"),names = ['ProteinID','PageRank'])
    pr = pd.read_csv(os.path.join(path,"ProteinID.csv"),names = ['ProteinID'])

    pra = pd.merge(pr,a,on='ProteinID',how='left')
    prab = pd.merge(pra,b,on='ProteinID',how='left')
    prabc = pd.merge(prab,c,on='ProteinID',how='left')
    prabcd = pd.merge(prabc,d,on='ProteinID',how='left')
    prabcde = pd.merge(prabcd,e,on='ProteinID',how='left')
    prabcdeh = pd.merge(prabcde,h,on='ProteinID',how='left')
    prabcdehl = pd.merge(prabcdeh,l,on='ProteinID',how='left')
    prabcdehlm = pd.merge(prabcdehl,m,on='ProteinID',how='left')
    prabcdehlmpr = pd.merge(prabcdehlm,pa,on='ProteinID',how='left')

    # Write to Output
    prabcdehlmpr.to_csv("allTopoFeatures.csv",index='ProteinID',sep=',')
    prabcdehlmpr.head()
# ArticulationPoints data is skewed
# Notes:
# 1. This main part requires sufficiently long time to run.
# 2. ArticulationPoints data is skewed
# 3. There are some remaining problem in calculating connectivity significance, including
#    getting result of "nan" and runtime error (and I have already excluded the case that
#    denominator is zero)
