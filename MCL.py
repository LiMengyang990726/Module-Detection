# Before running the following code, please do the following:
# pip install markov_clustering[drawing]
# pip install markov_clustering

import markov_clustering as mc
import networkx as nx
import random

# generate the graph that we are going to use
G = nx.read_edgelist('physical_edgelist_hprd')

# generate the corresponding adjacency matrix for the graph
matrix = nx.to_scipy_sparse_matrix(G)

# run MCL with default parameters
result = mc.run_mcl(matrix)
# get clusters
clusters = mc.get_clusters(result)

# visualize the final result
mc.draw_graph(matrix, clusters, pos=positions, node_size=50, with_labels=False, edge_color="silver")


# =================================================================================
# # parameterize: find inflation parameter that gives the largest modularity
# =================================================================================
for inflation in [i / 10 for i in range(15, 26)]:
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)
    Q = mc.modularity(matrix=result, clusters=clusters)
    print("inflation:", inflation, "modularity:", Q)
# then run this again, assume the best inflation value is 2.1
result = mc.run_mcl(matrix,inflation=2.1)
clusters = mc.get_clusters(result)
mc.draw_graph(matrix, clusters, pos=positions, node_size=50, with_labels=False, edge_color="silver")
