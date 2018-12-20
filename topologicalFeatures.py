import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import xlsxwriter
import subprocess as sp # for clear console: tmp = sp.call('clear',shell=True)
import csv

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

for i in range(len(diseases)):
    print(str(i)+" "+diseases[i])
    if(Gc.has_node(str(diseases[i]))):
        print("Added")
    else:
        diseases.pop(i)

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
    for j in range(len(diseases)):
        a += len(nx.shortest_path(Gc,source=str(i),target=str(diseases[j])))
    f_writer.writerow({'Gene_ID': i, 'Average Shortest Path to all Disease genes': a/float(len(diseases))})

f.close()


# ==============================================================================
#
# F2 local clustering coefficient
# rationale: the higher a node's clustering coefficient in each disease's network,
#            more likely the node belongs to this disease network
#
# ==============================================================================

neighbors_of_diseases = []

f= open("LocalCC.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'Local Clustering Coefficient']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

for i in range(len(diseases)):
    neighbors_of_diseases += [n for n in Gc.neighbors(diseases[i])]

for i in range(len(neighbors_of_diseases)):
    k = Gc.degree(neighbors_of_diseases[i])
    neighbors = [n for n in Gc.neighbors(neighbors_of_diseases[i])]
    N = 0
    for j in range(len(neighbors)):
        if (j+1) < len(neighbors):
            for k in range(j+1,len(neighbors)):
                if(Gc.has_edge(str(j),str(k))):
                    N = N+1
    denominator = k*(k-1)
    numerator = 2*N
    if (denominator != 0):
        localCC = float(numerator/denominator)
    else:
        localCC = 0 # lowest clustering coefficient
    f_writer.writerow({'Gene_ID': neighbors_of_diseases[i], 'Local Clustering Coefficient': localCC})
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

f= open("feature3456.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'DegreeCentrality','ClosenessCentrality','BetweennessCentrality','EigenvectoreCentrality','PageRank']
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

s4 = []
dic4 = nx.betweeness_centrality(Gc)
for key,value in dic4.items():
    s4 += [value]

s5 = []
dic5 = nx.eigenvector_centrality(Gc)
for key,value in dic5.items():
    s5 += [value]

s6 = []
dic6 = nx.percolation_centrality(Gc)
for key,value in dic6.items():
    s6 += [value]

s7 = []
dic7 = nx.pagerank(Gc)
for key,value in dic7.items():
    s7 += [value]

for i in len(s1):
    f_writer.writerow({'Gene_ID': s1[i], 'DegreeCentrality': s2[i], 'ClosenessCentrality': s3[i],
                        'BetweennessCentrality': s4[i], 'EigenvectoreCentrality': s5[i],
                        'PercolationCentrality': s6[i], 'PageRank': s7[i]})

f.close()


# ==============================================================================
#
# connectivity significance
#
# ==============================================================================

from scipy.special import comb

f= open("ConnectivitySignificance.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
fieldnames = ['Gene_ID', 'ConnectivitySignificance']
f_writer = csv.DictWriter(f, fieldnames=fieldnames)

alpha = 2
for node in Gc.nodes():
    neighbors = set(Gc.neighbors(node))

    k = Gc.degree(node)

    ks = 0
    for neighbor in neighbors:
        if neighbor in diseases:
            ks += 1

    ks0 = ks + (alpha-1)*ks

    s = len(diseases)

    s0 = s + (alpha-1)*s

    n1 = comb(s0,ks0)                               # numerator 1
    n2 = comb((N-s),(k-ks))                         # numerator 2
    d = comb((N+(alpha-1)*ks),(k+(alpha-1)*ks))     # denominator
    if (d == 0):
        pvalue = -1                                 # set those invalid number to nan, meaning those nodes doesn't have connection with disease node
    else:
        pvalue = (n1*n2)/d
    f_writer.writerow({'Gene_ID': node, 'ConnectivitySignificance': pvalue})

# Remaining problem:
# There are results "nan" and "runtime error" problem unsolved


# ==============================================================================
#
# articulation protein
#
# ==============================================================================

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


    def APUtil(self,u, visited, ap, parent, low, disc):

        children = 0

        visited[int(u)] = True

        self.Time += 1

        # recursive all neighbors of this node
        for v in test.graph.neighbors(str(u)):
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if visited[int(v)] == False:
                parent[int(v)] = int(u)
                children += 1
                self.APUtil(v, visited, ap, parent, low, disc)

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

        f= open("articulationPoints.csv",mode='w') # Only the first level neighbors of disease nodes are shown here
        fieldnames = ['Gene_ID', 'isArticulationPoint']
        f_writer = csv.DictWriter(f, fieldnames=fieldnames)

        largestIndex = largestIndex()
        visited = [False] * (largestIndex)
        disc = [float("Inf")] * (largestIndex)
        low = [float("Inf")] * (largestIndex)
        parent = [-1] * (largestIndex)
        ap = [-1] * (largestIndex)           # To store articulation points, 1 represents True, 0 represents False
                                             # -1 represents those indices that don't have a node corresponding to it
        counter = 0
        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in self.graph.nodes:
            if visited[counter] == False:
                self.APUtil(i, visited, ap, parent, low, disc)
            counter += 1

articulation = articulationGraph(Gc)
articulation.AP()

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
        if((j != i) and (j in diseases)): # exclude itself, and only include the modularity with disease nodes
            temp += B[counter1,counter2]
            counter3 += 1
        counter2 += 1

    counter1 += 1

    temp = float(temp/counter3)
    a += [temp]
    f_writer.writerow({'Gene_ID': i, 'isArticulationPoint': temp})

# Notes:
# 1. Due to the computational power limit and restraint of logging in through ssh,
#    all the above computations are only partially done. Please run the whole script
#    to get all output files.
# 2. Networkx's percolation_centrality package has some fallacy, and I might need
#    some time to fix it.
# 3. Networkx doesn't have function to calculate markov centrality, and I am confused
#    by the difference between markov centrality, random walk betweenness, and pagerank.
#    If neccessary, I will try to implement the calculation of random walk betweennes
# 4. I haven't finish the implementation of converting modularity_matrix to a series,
#    and haven't finishe the implementation of finding articulation protein.
# Questions:
# 1. Shall we convert all proteins represented in Uniprot and GO term form
#    into geneID representation?
# 2. If we are going to use geneID representation, geneID is generic to an individual
#    while it doesn't have topological properties
