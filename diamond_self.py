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
# Prepare all math calculations
# =============================================================================
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
        # print (prob)                                                           
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
    # Setting initial set of nodes not in cluster
    # ------------------------------------------------------------------
    for node in cluster_nodes:
        not_in_cluster |= neighbors[node]
    not_in_cluster -= cluster_nodes


    # ------------------------------------------------------------------
    #
    # M A I N     L O O P 
    #
    # ------------------------------------------------------------------

    all_p = {}

    while len(added_nodes) < X:    

        # ------------------------------------------------------------------
        #
        # Going through all nodes that are not in the cluster yet and
        # record k, kb and p 
        #
        # ------------------------------------------------------------------
        
        info = {}
    
        pmin = 10
        next_node = 'nix'
        reduced_not_in_cluster = reduce_not_in_cluster_nodes(all_degrees,
                                                             neighbors,G,
                                                             not_in_cluster,
                                                             cluster_nodes,alpha)
        
        for node,kbk in reduced_not_in_cluster.iteritems():
            # Getting the p-value of this kb,k
            # combination and save it in all_p, so computing it only once!
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
    
            info[node] = (k,kb,p)

        # ---------------------------------------------------------------------
        # Adding node with smallest p-value to the list of aaglomerated nodes
        # ---------------------------------------------------------------------
        added_nodes.append((next_node,
                            info[next_node][0],
                            info[next_node][1],
                            info[next_node][2]))

        # Updating the list of cluster nodes and s0
        cluster_nodes.add(next_node)
        s0 = len(cluster_nodes)
        not_in_cluster |= ( neighbors[next_node] - cluster_nodes )
        not_in_cluster.remove(next_node)

    return added_nodes

# ===========================================================================
#
#   M A I N    D I A M O n D    A L G O R I T H M
# 
# ===========================================================================
def DIAMOnD(G_original,seed_genes,max_number_of_added_nodes,alpha,outfile = None):
    # 1. throwing away the seed genes that are not in the network
    all_genes_in_network = set(G_original.nodes())
    seed_genes = set(seed_genes)
    disease_genes = seed_genes & all_genes_in_network

    if len(disease_genes) != len(seed_genes):
        print "DIAMOnD(): ignoring %s of %s seed genes that are not in the network" %(
            len(seed_genes - all_genes_in_network), len(seed_genes))
   


 
    # 2. agglomeration algorithm. 
    added_nodes = diamond_iteration_of_first_X_nodes(G_original,
                                                     disease_genes,
                                                     max_number_of_added_nodes,alpha)
    # 3. saving the results 
    with open(outfile,'w') as fout:
        print>>fout,'\t'.join(['#rank','DIAMOnD_node'])
        rank = 0
        for DIAMOnD_node_info in added_nodes:
            rank += 1
            DIAMOnD_node = DIAMOnD_node_info[0]
            p = float(DIAMOnD_node_info[3])
            print>>fout,'\t'.join(map(str,([rank,DIAMOnD_node])))

    return added_nodes


# ===========================================================================
#
#   R U N N I N G   D I A M O N D    A L G O R I T H M
# 
# ===========================================================================
if __name__ == '__main__':

    G_original = nx.read_edgelist("./gene-network.tsv")

    seed_genes = []
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
    for key in disease_dic.keys():
        seed_genes = Union(seed_genes,disease_dic[key])

    max_number_of_added_nodes = 200
    alpha = 2
    outfile_name = "duplicateExperiment"
    added_nodes = DIAMOnD(G_original,
                          seed_genes,
                          max_number_of_added_nodes,alpha,
                          outfile=outfile_name)
    
    print "\n results have been saved to '%s' \n" %outfile_name
