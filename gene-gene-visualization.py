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
nx.draw(G.node_size=1)
plt.show()
plt.savefig("Gene-network-overview.png")

disease_dic = {}
disease_file  = open("gene-disease0.TSV", 'r')

# ===============================================================
# # change the input style
# ===============================================================
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
# # plot all rest nodes from gene-networkx except seed genes
# ===============================================================
all_rest_nodes = list(G) # at the end should be 
for key in disease_dic.keys():
    temp = [list(map(int, x)) for x in disease_dic[key]]
    all_rest_nodes = [x for x in all_rest_nodes if x not in temp]
nx.draw_networkx_nodes(G,pos,
                        nodelist= all_rest_nodes,
                        node_size=5,
                        node_color='gray',
                        alpha=0.8)
nx.draw_networkx_edges(G,pos,
                       edgelist=list(G.edges()),
                       width=8,alpha=0.5,edge_color='r')

# ===============================================================
# # plot all seed genes
# ===============================================================
c = plt.get_cmap("plasma")
for key in disease_dic.keys(): 
    nx.draw_networkx_nodes(G,pos,
                            nodelist=disease_dic[key],
                            node_size=5,
                            cmap = c)

plt.axis('off')
plt.savefig("gene-visualization.png")
plt.show()
