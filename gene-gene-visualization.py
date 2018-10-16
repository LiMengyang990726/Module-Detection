# import networkx as nx
# import matplotlib.pyplot as plt
#
# G = nx.read_edgelist("./gene-network.tsv",comments = "#", nodetype = int)
# pos = nx.spring_layout(G)

disease_dic = {}
disease_file  = open("gene-disease0.TSV", 'r')

# change the file into a dictionary
for line in disease_file:
    li = line.strip()
    if not li.startswith("#"):
        li2 = li.split(' ',1)
        disease_key = li2[0]
        print ("the key is: "+disease_key)
        disease_list = [l for l in (li2[1]).split('/')]
        print (disease_list)
        disease_dic.update({disease_key: disease_list})

# plot the network with color
# if they are in the same community, same color
# if not covered by any community, grey
for key in disease_dic.keys():
    nx.draw_networkx_nodes(G,pos,
                           nodelist=disease_dic[key],
                           node_color=range(70),
                           node_size=500,
                       alpha=0.8,
                       cmap=plt.cm.plasma)
