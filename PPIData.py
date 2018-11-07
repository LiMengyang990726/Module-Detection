import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import subprocess as sp # for clear console: tmp = sp.call('clear',shell=True)

# ===============================================================
# Input
# ===============================================================
G = nx.read_edgelist("./gene-network.tsv")
Gc = max(nx.connected_component_subgraphs(G), key=len)

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

# ===============================================================
# Network Analysis
# ===============================================================
f= open("Network_Analysis.txt","w+")

f.write(nx.info(G))
f.write("\nAverage clustering coefficient " + repr(nx.average_clustering(G)))

plt.hist(list(dict(nx.degree(G)).values()))
plt.title("Degree Histogram")
plt.savefig("Degree_distribution.png")

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
f.write("\nLargest degree " + str(degree_sequence[0]))

f.write("\nThe largest connected component ")
f.write("\n\t\tNumber of nodes " + str(len(Gc)))
f.write("\n\t\tNumber of edges " + str(Gc.number_of_edges()))
f.write("\n\t\tAverage shortest path length " + str(nx.average_shortest_path_length(Gc)))
f1 = open("Edge_Betweenness.txt","w+")
# f1.write(nx.edge_betweenness_centrality(Gc))
# for line in f1:

f.close()

# ===============================================================
# Percentage Analysis
# ===============================================================
import xlsxwriter

f1 = xlsxwriter.Workbook('Percentage_of_Disease_Gene_in_PPI.xlsx')
sh = f1.add_worksheet()

sh.write(0,0,'Disease Name')
sh.write(0,1,'Number of disease genes')
sh.write(0,2,'Number of disease genes inside LCC')
sh.write(0,3,'Percentage of number of disease genes')

row = 1
col = 0
count = 0
cell_format1 = f1.add_format()
cell_format2 = f1.add_format()
cell_format1.set_num_format('0')
cell_format2.set_num_format('0.00000000000')
for i in disease_dic:
    sh.write_string(row,col,i)
    temp = 0
    length = len(disease_dic[i])
    for j in disease_dic[i]:
        if j in list(Gc):
            temp+=1
    sh.write_number(row,col+1,length,cell_format1)
    sh.write_number(row,col+2,temp,cell_format1)
    sh.write_number(row,col+3,(temp*1.0)/length,cell_format2)
    row += 1

f1.close()

# ===============================================================
# Convert Data Into Features
# ===============================================================
for key in disease_dic:
    l = len(disease_dic[key])
    nodes = []
    for i in range(l):
        if disease_dic[key][i] in Gc.nodes:
            nodes.append(disease_dic[key][i])
    sub = Gc.subgraph(nodes)
    n = nx.number_connected_components(sub)
    print("number of connected components is " + str(n) + "\n")
