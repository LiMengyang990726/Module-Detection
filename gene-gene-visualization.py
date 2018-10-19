import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as col
import warnings
# for filer the numpy binary imcompatibilty warning in linux server
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

G = nx.read_edgelist("./gene-network.tsv",comments = "#", nodetype = int)
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
i = 10 # random starting point for the color changing
for key in disease_dic.keys(): # draw the hightlighted nodes
    nx.draw_networkx_nodes(G,pos,
                           nodelist=disease_dic[key],
                           node_size=500,
                           node_color=col.Colormap(plasma,N=i),
                           edge_color=col.Colormap(plasma,N=i),
                           # cmap=plt.cm.plasma)
                           )
    i++
# draw those that aren't mentioned
all_nodes = G.nodes()
for key in disease_dic.keys():
    for i in disease_dic[key]:
        all_nodes.remove(disease_dic[key][i])
nx.draw_networkx_nodes(G,pos,
                       nodelist= all_nodes,
                       node_size=500,
                       node_color='gray',
                       alpha=0.8)
