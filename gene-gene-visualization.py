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
Gc = max(nx.connected_component_subgraphs(G), key=len)
pos = nx.spring_layout(Gc)

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
all_rest_nodes = list(Gc) # at the end should be
for key in disease_dic.keys():
    temp = [list(map(int, x)) for x in disease_dic[key]]
    all_rest_nodes = [x for x in all_rest_nodes if x not in temp]

nx.draw_networkx_nodes(Gc,pos,
                        nodelist= all_rest_nodes,
                        node_size=5,
                        node_color='black',
                        alpha=0.8)
nx.draw_networkx_edges(Gc,pos,
                       edgelist=list(Gc.edges()),
                       width=1,edge_color='gray')

# ===============================================================
# # plot single color for one disease
# ===============================================================

Gtemp = nx.Graph()
len(disease_dic["adrenalglanddiseases"]) # 18
for i in range(17):
    for x in xrange(i,17):
    Gtemp.add_edge(disease_dic["adrenalglanddiseases"][i],nx.common_neighbors(Gc,disease_dic["adrenalglanddiseases"][i],disease_dic["adrenalglanddiseases"][x]))

posTemp = nx.spring_layout(Gtemp)

nx.draw_networkx_nodes(Gtemp,posTemp,
                        nodelist= Gtemp.nodes,
                        node_size=5,
                        node_color='black',
                        alpha=0.8)
nx.draw_networkx_edges(Gtemp,posTemp,
                       edgelist=list(Gtemp.edges()),
                       width=1,edge_color='gray')

# disease_dic["adrenalglanddiseases"].remove('1187')

nx.draw_networkx_nodes(Gtemp,posTemp,
                    nodelist=disease_dic["adrenalglanddiseases"],
                    node_size=8,
                    node_color='red')
plt.axis('off')
plt.savefig("adrenalglanddiseases.png")

# ===============================================================
# # plot all seed genes in one color first
# ===============================================================
for key in disease_dic.keys():
    try:
        nx.draw_networkx_nodes(Gc,pos,
                            nodelist=disease_dic[key],
                            node_size=5,
                            node_color='red')
    except:
        print ("node not in pos")
plt.axis('off')
plt.savefig("gene-visualization-single-color.png")
plt.show()
# ===============================================================
# # plot all seed genes
# ===============================================================
import random
def colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = float((int(r) % 256)/256)
        g = float((int(g) % 256)/256)
        b = float((int(b) % 256)/256)
        ret.append((r,g,b))
    return ret

for key in disease_dic.keys():
    try:
        nx.draw_networkx_nodes(Gc,pos,
                            nodelist=disease_dic[key],
                            node_size=5,
                            node_color=colors)
    except:
        print ("node not in pos")

plt.axis('off')
plt.savefig("gene-visualization-multiple-color.png")
plt.show()
