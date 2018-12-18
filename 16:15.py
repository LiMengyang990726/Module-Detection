import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
import xlsxwriter
import subprocess as sp # for clear console: tmp = sp.call('clear',shell=True)

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

# F1 average shortest path length:
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
f.close()

# F2 local clustering coefficient
# rationale: the higher a node's clustering coefficient in each disease's network,
#            more likely the node belongs to this disease network
neighbors_of_diseases = []
f= open("LocalCC.txt","w+")
f.write("Only the first level neighbors of disease nodes are shown here\n")
f.write("Gene_ID      Local Clustering Coefficient\n")

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
    f.write(str(neighbors_of_diseases[i])+ "   "+str(localCC)+"\n")
f.close()

# F3 degree centrality
# rationale: Higher, more likely to be involved in a more important functional module

# F4 closeness centrality
# rationale: Higher, more functionally important as needs to be communicated quickly.

# F5 betweeness centrality
# rationale: An important node will lie on a higher proportion of the paths.

# F6 eigenvector centrality

import xlsxwriter
def output(Gc):
    workbook = xlsxwriter.Workbook('feature3456.xlsx')
    worksheet = workbook.add_worksheet()

    row = 0
    col = 0

    worksheet.write(row, col,'GeneID')
    worksheet.write(row, col+1,'DegreeCentrality')
    worksheet.write(row, col+2,'ClosenessCentrality')
    worksheet.write(row, col+3,'EigenvectoreCentrality')
    worksheet.write(row, col+4,'MarkovChainCentrality')

    row += 1

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
    dic4 = nx.eigenvector_centrality(Gc)
    for key,value in dic4.items():
        s4 += [value]

    for i in len(s1):
        worksheet.write(row, col,s1[i])
        worksheet.write(row, col+1,s2[i])
        worksheet.write(row, col+2,s3[i])
        worksheet.write(row, col+3,s4[i])
        row += 1

    workbook.close()
output(Gc)

# markov chain centraility
# rationale: see the proportion of each gene's importance)


# connectivity significance

# articulation protein
class articulationGraph:

    def __init__(self,networkxGraph):
        self.graph =  networkxGraph
        self.Time = 0

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
                    ap[int(u)] = True

                #(2) If u is not root and low value of one of its child is more
                # than discovery value of u.
                if parent[int(u)] != -1 and low[int(v)] >= disc[int(u)]:
                    ap[int(u)] = True

                # Update low value of u for parent function calls
            elif int(v) != parent[int(u)]:
                low[int(u)] = min(low[int(u)], disc[int(v)])


    #The function to do DFS traversal. It uses recursive APUtil()
    def AP(self):

        visited = [False] * (len(self.graph.nodes))
        disc = [float("Inf")] * (len(self.graph.nodes))
        low = [float("Inf")] * (len(self.graph.nodes))
        parent = [-1] * (len(self.graph.nodes))
        ap = [False] * (len(self.graph.nodes)) #To store articulation points
        counter = 0
        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in self.graph.nodes:
            if visited[counter] == False:
                self.APUtil(i, visited, ap, parent, low, disc)
            counter += 1

        for index, value in enumerate (ap):
            if value == True:
                print(index)

# modularity
