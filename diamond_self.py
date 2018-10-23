import time
import cPickle
import networkx as nx
import numpy as np
import copy
import scipy.stats
from collections import defaultdict
import csv
import sys

# =============================================================================
def check_input_style(input_list):
    try:
        network_edgelist_file = gene-networkx[1]
        seeds_file = input_list[2]
        max_number_of_added_nodes = int(input_list[3])
    # if no input is given, print out a usage message and exit
    except:
        print_usage()
        sys.exit(0)
        return 

# =============================================================================
# Prepare all math calculations
# ================================================================================
def compute_all_gamma_ln(N):
    """
    precomputes all logarithmic gammas 
    """
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
    """                                                                        
    -------------------------------------------------------------------        
    Computes the p-value for a node that has kb out of k links to              
    seeds, given that there's a total of s sees in a network of N nodes.       
                                                                               
    p-val = \sum_{n=kb}^{k} HypergemetricPDF(n,k,N,s)                          
    -------------------------------------------------------------------        
    """                                                                        
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

# =============================================================================
# Reduce number of calculations
# =============================================================================
def reduce_not_in_cluster_nodes(all_degrees,neighbors,G,not_in_cluster,cluster_nodes,alpha): 
    reduced_not_in_cluster = {}                                                        
    kb2k = defaultdict(dict)                                                           
    for node in not_in_cluster:                                                        
        
        k = all_degrees[node]                                                          
        kb = 0                                                                         
        # Going through all neighbors and counting the number of module neighbors        
        for neighbor in neighbors[node]:                                               
            if neighbor in cluster_nodes:                                              
                kb += 1
        
        #adding wights to the the edges connected to seeds
        k += (alpha-1)*kb
        kb += (alpha-1)*kb
        kb2k[kb][k] =node

    # Going to choose the node with largest kb, given k                                
    k2kb = defaultdict(dict)                                                           
    for kb,k2node in kb2k.iteritems():                                                 
        min_k = min(k2node.keys())                                                     
        node = k2node[min_k]                                                           
        k2kb[min_k][kb] = node                                                         
                                                                                       
    for k,kb2node in k2kb.iteritems():                                                 
        max_kb = max(kb2node.keys())                                                   
        node = kb2node[max_kb]                                                         
        reduced_not_in_cluster[node] =(max_kb,k)                                       
                                                                                       
    return reduced_not_in_cluster                                                      

#======================================================================================
#   C O R E    A L G O R I T H M
#======================================================================================
def diamond_iteration_of_first_X_nodes(G,S,X,alpha):
    