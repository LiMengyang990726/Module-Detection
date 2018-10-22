import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as col
import warnings
import sys
# for filer the numpy binary imcompatibilty warning in linux server
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


G = nx.read_edgelist("./gene-network.tsv")
pos = nx.spring_layout(G)

disease_dic = {}
disease_file  = open("gene-disease0.TSV", 'r')

# change the file into a dictionary
for line in disease_file:
    li = line.strip()
    if not li.startswith("#"):
        li2 = li.split(' ',1)
        disease_key = li2[0]
        print ("the key is: "+disease_key)
        disease_list = [l for l in (li2[1]).split('/')] # should add them to a set of seed gene
        print (disease_list)
        disease_dic.update({disease_key: disease_list})

# plot the network with color
# if they are in the same community, same color
# if not covered by any community, grey

c = plt.get_cmap("plasma")
all_nodes = G.nodes()
# as we found that gene-network lists didn't contain all the nodes

f = open('all_nodes.tsv', 'w')
sys.stdout = f
for key in disease_dic.keys():
    all_nodes = list(set(disease_dic[key]) | set(all_nodes)) # get all nodes, however still have problem in finding some nodes
allLength = len(all_nodes)
c = 0
for c in range(allLength):
    f.write(all_nodes[allLength])
    f.write("\n")
f.close()

# removing the mentioned nodes from the
for key in disease_dic.keys():
    length = len(disease_dic[key])
    tempList = disease_dic[key]
    i = 0
    for i in range(length):
        try:
            all_nodes.remove(tempList[i])
        except:
            print ("node %d is not in the list" % int(tempList[i]))

for key in disease_dic.keys(): # draw the hightlighted nodes
    try:
        nx.draw_networkx_nodes(G,pos,
                           nodelist=disease_dic[key],
                           node_size=50,
                           cmap = c)
        nx.draw_networkx_nodes(G,pos,
                       nodelist= all_nodes,
                       node_size=50,
                       node_color='gray',
                       alpha=0.8)
        plt.axis('off')
        plt.savefig("gene-visualization.png") # save as png
        plt.show()
    except:
        print ("the node has some error but I want to ignore it first")
