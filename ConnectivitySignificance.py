
# THIS ONE DOESN'T WORK FOR NOW
# If use the origical DIAMOnD algorithm, we will get the error message
# 'collections.defaultdict' object has no attribute 'iteritems'
# in code "for node,kbk in reduced_not_in_cluster.iteritems():"
# (line 332 in diamond.py file)


import time
import networkx as nx
import numpy as np
import copy
import scipy.stats
from collections import defaultdict
import csv
import sys


# =============================================================================
def read_input(network_file,seed_file):
    sniffer = csv.Sniffer()
    line_delimiter = None
    for line in open(network_file,'r'):
        if line[0]=='#':
            continue
        else:
            dialect = sniffer.sniff(line)
            line_delimiter = dialect.delimiter
            break
    if line_delimiter == None:
        print("network_file format not correct")
        sys.exit(0)
    # read the network:
    G = nx.Graph()
    for line in open(network_file,'r'):
        # lines starting with '#' will be ignored
        if line[0]=='#':
            continue
        # The first two columns in the line will be interpreted as an
        # interaction gene1 <=> gene2
        #line_data   = line.strip().split('\t')
        line_data = line.strip().split(line_delimiter)
        node1 = line_data[0]
        node2 = line_data[1]
        G.add_edge(node1,node2)
    # read the seed genes:
    seed_genes = set()
    for line in open(seed_file,'r'):
        # lines starting with '#' will be ignored
        if line[0]=='#':
            continue
        # the first column in the line will be interpreted as a seed
        # gene:
        line_data = line.strip().split('\t')
        seed_gene = line_data[0]
        seed_genes.add(seed_gene)
    return G,seed_genes


# ================================================================================
def compute_all_gamma_ln(N):
    gamma_ln = {}
    for i in range(1,N+1):
        gamma_ln[i] = scipy.special.gammaln(i)
    return gamma_ln

# =============================================================================
def logchoose(n, k, gamma_ln):
    if n-k+1 <= 0:
        return scipy.infty
    lgn1  = gamma_ln[n+1]
    lgk1  = gamma_ln[k+1]
    lgnk1 = gamma_ln[n-k+1]
    return lgn1 - [lgnk1 + lgk1]

# =============================================================================
def gauss_hypergeom(x, r, b, n, gamma_ln):
    return np.exp(logchoose(r, x, gamma_ln) +
                  logchoose(b, n-x, gamma_ln) -
                  logchoose(r+b, n, gamma_ln))

# =============================================================================
def pvalue(kb, k, N, s, gamma_ln):
    p = 0.0
    for n in range(kb,k+1):
        if n > s:
            break
        prob = gauss_hypergeom(n, s, N-s, k, gamma_ln)
        # print prob
        p += prob
    if p > 1:
        return 1
    else:
        return p

# =============================================================================
def get_neighbors_and_degrees(G):
    neighbors,all_degrees = {},{}
    for node in G.nodes():
        nn = set(G.neighbors(node))
        neighbors[node] = nn
        all_degrees[node] = G.degree(node)
    return neighbors,all_degrees


#======================================================================================
#   C O R E    A L G O R I T H M
#======================================================================================
def diamond_iteration_of_first_X_nodes(G,S,alpha):
    N = G.number_of_nodes()
    added_nodes = []
    # ------------------------------------------------------------------
    # Setting up dictionaries with all neighbor lists
    # and all degrees
    # ------------------------------------------------------------------
    neighbors,all_degrees = get_neighbors_and_degrees(G)
    # ------------------------------------------------------------------
    # Setting up initial set of nodes in cluster
    # ------------------------------------------------------------------
    cluster_nodes = set(S)
    not_in_cluster = set()
    s0 = len(cluster_nodes)
    s0 += (alpha-1)*s0
    N +=(alpha-1)*s0
    # ------------------------------------------------------------------
    # precompute the logarithmic gamma functions
    # ------------------------------------------------------------------
    gamma_ln = compute_all_gamma_ln(N+1)
    # ------------------------------------------------------------------
    #
    # M A I N     L O O P
    #
    # ------------------------------------------------------------------
    all_p = {}
    for i in Gc.nodes:
        pmin = 10
        next_node = 'nix'
        kb,k = kbk
        try:
            p = all_p[(k,kb,s0)]
        except KeyError:
            p = pvalue(kb, k, N, s0, gamma_ln)
            all_p[(k,kb,s0)] = p
        # recording the node with smallest p-value
        if p < pmin:
            pmin = p
            next_node = node
    return all_p

if __name__ == '__main__':

    # read the network and the seed genes:
    G_original,seed_genes = read_input("gene-network.tsv","gene-disease0.TSV")

    alpha = 2
    pValues = diamond_iteration_of_first_X_nodes(G_original,seed_genes,alpha)
