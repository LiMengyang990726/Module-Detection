import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import subprocess as sp # for clear console: tmp = sp.call('clear',shell=True)

# ===============================================================
# Network Analysis
# ===============================================================
G = nx.read_edgelist("./gene-network.tsv")
f= open("Network_Analysis.txt","w+")
f.write(nx.info(G))

f.write("\nAverage clustering coefficient " + repr(nx.average_clustering(G)))

plt.hist(list(dict(nx.degree(G)).values()))
plt.title("Degree Histogram")
plt.savefig("Degree_distribution.png")

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
f.write("\nLargest degree " + str(degree_sequence[0]))

Gc = max(nx.connected_component_subgraphs(G), key=len)
f.write("\nThe largest connected component ")
f.write("\n\t\tNumber of nodes " + str(len(Gc)))
f.write("\n\t\tNumber of edges " + str(Gc.number_of_edges()))
f.write("\n\t\tAverage shortest path length " + str(nx.average_shortest_path_length(Gc)))
f1 = open("Edge_Betweenness.txt","w+")
f1.write(nx.edge_betweenness_centrality(Gc))
for line in f1:


f.close()

# ===============================================================
# Network Analysis
# ===============================================================
disease_dic = {}
disease_file  = open("gene-disease0.TSV", 'r')
for line in disease_file:
    li = line.strip()
    if not li.startswith("#"):
        li2 = li.split(' ',1)
        disease_key = li2[0]
        print ("the key is: "+disease_key)
        disease_list = [l for l in (li2[1]).split('/')]
        print (disease_list)
        disease_dic.update({disease_key: disease_list})

D = []
for i in disease_dic:
    for j in disease_dic[i]:
        temp = 0
        if j in list(Gc):
            temp += 1
        D.append(temp)
