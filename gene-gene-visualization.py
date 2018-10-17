import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as col

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
for key in disease_dic.keys():
    nx.draw_networkx_nodes(G,pos,
                           nodelist=disease_dic[key],
                           node_size=500,
                           node_color=col.Colormap(plasma,N=i)
                           alpha=0.8,
                           # cmap=plt.cm.plasma)
                           )
    i++

# find those nodes that should be added to each network according to their connectivity significance in the first round
for key in disease_dic.keys():
    thisList = disease_dic[key]


# gamma(z) = (z-1)!
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
