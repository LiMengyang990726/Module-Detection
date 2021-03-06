-------------------------------
## Title

"Predicting disease modules using machine learning based Approach"

## Introduction
- Purpose of the research:
  - Evaluate the Diamond Disease Module Detection algorithm
  - Use machine learning approach to detect the module in a network

## Background
I)  Network Topological properties of network：[1]

Understanding Topological properties is essential to understand the previous algorithms for detecting modules

  - Local:
    - Clustering coefficient: **Fraction of interconnections**

    Star network/A highly centrailized network has a clustering coefficient of 0.  

    Let v be a node, and k<sub>v</sub> denotes v's degree, and N<sub>v</sub> denotes number of links between neighbors of v.

    CC(v) = 2N<sub>v</sub>/(k<sub>v</sub> * (k<sub>v</sub>-1))

    - Shortest path: **Longer the shortest path distance from the seed node, less relevant**

    - Centrality: **Identify the most important nodes in a graph**

    Betweenness is a centrality measure, and it had been used in the Divisive Link Community algorithm.

    - Modularity: **Measure how well is it to group the network in certain way. Represents the possibility difference with the observed edges and the expected edges.**


    The original formula was introduced in this [paper](https://arxiv.org/pdf/cond-mat/0308217.pdf).

    Then there are some modified forms introduced [here](https://en.wikipedia.org/wiki/Modularity_(networks)).

    Derivation between the above two forms are shown [here](https://arxiv.org/pdf/cond-mat/0408187.pdf)
  - Global
    - Modularity
    - Clustering coefficient

    CC(G) = avg<sub>v in G</sub>(CC(v))

II) Previous Algorithm

There are handful of module detection algorithms in the previous studies, and some of the most commonly used ones are summarized as below

  - Modularity based
      - Louvain algorithm [2]
      - Greedy optimization [3]
      - Divisive Link community [4]
  - Random flow based
      - MCL [5]
  - Most recent
      - Diamond (based on connectivity significance)
      - this paper proposed a machine learning approach

III) Comparison for the previous work (use a table)


IV) Types of Modules in molecular networks

Specifically for protein-protein network, or generally for a biological network (regulatory and metabolic networks etc.), the following three types of modules should be taken care.

  - Topological modules

  Can be found through: Compute and determine certain thing through the calculation of one or more topological network properties

  - Functional modules

  A functional module (5) is defined as a group of genes or their products which are related by one or more genetic or cellular interactions.

  A functional module has more relations between themselves rather than with others.


  Integrative(meaning independent but biologically related data sets or different genetic and molecular networks) data analysis has been performed with correlation mapping and mean Pearson correlations.

  [Read more](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC275479/#gkg838c5)

  - Disease modules



## Material and Methods

## Results and Discussion

## Conclusion

## References:

[1] Mason, O., & Verwoerd, M. (2007). Graph theory and networks in biology. IET systems biology, 1(2), 89-119.

[2] Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. Journal of statistical mechanics: theory and experiment, 2008(10), P10008.

[3] Clauset, A., Newman, M. E., & Moore, C. (2004). Finding community structure in very large networks. Physical review E, 70(6), 066111.

[4] Newman, M. E., & Girvan, M. (2004). Finding and evaluating community structure in networks. Physical review E, 69(2), 026113.

[5] Van Dongen, S. M. (2000). Graph clustering by flow simulation (Doctoral dissertation).

[6]Hartwell L.H., Hopfield,J.J., Leibler,S. and Murray,A.W. (1999) From molecular to modular cell biology. Nature, 402 (suppl.), C47–C52

[7] Tornow, S., & Mewes, H. W. (2003). Functional modules by relating protein interaction networks and gene expression. Nucleic acids research, 31(21), 6283-9.

-------------------------------
**Question 1:
  How to Reference those files that are in ppt format, youtube or github link?**
**Question 2:
  If the paper refered to other papers, how should we write the refercen?**

-------------------------------------------------

# Journal
## Network basics
* **Small world property**: the average path length is usually log(n) , n is the number of nodes. [Details](https://en.wikipedia.org/wiki/Small-world_network)
* **High clustering coefficient**

## Mathematical Preliminary (Details refer to [this](https://arxiv.org/pdf/q-bio/0604006.pdf))
* **Shortest path / Geodesic distance**: δ(u, v)

    In our case, we are using unweighted graph. Commonly used instructions: `nx.shortest_path_length[G,source,target]`,`nx.all_shortest_path[G,source,target]`. [Please refer to this for the full shortest-path documentation](https://networkx.github.io/documentation/stable/reference/algorithms/shortest_paths.html)
* **Clustering coefficient**
* **Degree distribution**: in scale-free networks, most nodes have relatively low degrees. Those nodes with unusually high degrees are called hubs.
                       when building degree distribution, the following mathematical models are used:
                       1. Classic model
                       2. Barabasi-ALbert Model
                       3. Duplication-Divergence Model
* **Modularity (Q)**: Represents the possiblity difference with the observed edges and the expected edges.
              The original formula was introducted in this [paper](https://arxiv.org/pdf/cond-mat/0308217.pdf).
              Then there are some modified forms introduced [here](https://en.wikipedia.org/wiki/Modularity_(networks)).
              Derivation between the above two forms are shown [here](https://arxiv.org/pdf/cond-mat/0408187.pdf)
              Another matric form is introduced as the picture.
* **Centraility**: a significant measurement to find out the most important nodes
               Have three types: Degree centraility, Closeness centraility, and Betweeness centraility
* **Markov Centraility**: a concept that reflects the notion of how central a node *t* is in a network relative to a particular node *r* or all nodes *R*. [For more details, please refer to this.](http://datamining.dongguk.ac.kr/graph/p266-white.pdf) The importance that node *t* relative to root node(s) *R* is denoted:

  I(t|R) = 1/( (1/|R|) * Σr∈R mrt )

  where mrt = Σ (end:∞ from:n=1) nf(n)

  f(n) is the probalibity of transitioning from node i to node j. If there is an edge between node i and node j, f(n) = 1/doutj; Otherwise, f(n) = 0

  The I(t|R) formula is derived from its orginal definition formula:

  I(t|r) = Σ (end: all paths from t to r; from the first path) λ to the power of -|ith path length|

  Here, we choose the path as K-short node-disjoint path.

  A random walk is an example of  **Markov Chain** using the transition matrix.

## Module detection algorithm
* **Agglomerative** Bottom-up
    1. Hierarchical clustering:

    Expressed in the Dendrogram form.

    Merge pair of clusters that are closest. [Please refer to this for more details.](https://www.youtube.com/watch?v=XJ3194AmH40) [Or please refer to this](https://www.displayr.com/what-is-hierarchical-clustering/)

    Disadvantage: Tendency to find only the cores of community while leave out the periphery

    2. Greedy way:

    Optimize modularity approach.

    Its time complexity is O(mdlogn) with d is the depth of the dendrogram.

    Start with each vertex being the sole member of a community of one, then repeatedly join together the two communities whoes amalgamation produces the largest increase in Q. After n-1 steps, we have **only one (the difference with Louvain algorithm)** community.
                   [Please refer to this for more details.](https://arxiv.org/pdf/cond-mat/0408187.pdf)

    **Algorithm details:**

     i) Use a *matrix* to store ΔQij

     ∆Qij =1/2m − kikj/(2m)² if i, j are connected. 0 otherwise (as it is going to be sparse).

     Each row is represented as both a *balanced binary tree*(insertion/searching can be down in O(logn)time) and a *max heap*(largest ∆Qij can be found in linear time)

     An *array* to store aij=ki/2m

     ii) Select the largest ∆Qij from the matrix. Assume if i and j join together, we name the community as j community. Then we *only need to update the jth row and column, delete ith column*.

     The updating(will benefit from balanced binary search tree) jth row and column (exactly same to row, we use row as example here) is according to this:

     If k is connected to i and j: new ∆Qjk = ∆Qik+∆Qij

     If k is connected to i not j: new ∆Qjk = ∆Qik-2ajak

     If k is connected to j not i: new ∆Qjk = ∆Qjk-2aiak

     iii) Repeat ii) until only one community remains.

     Disadvantage: Using max heap can help us find the largest value quickly, it also requires more effort to maintain it, means reform the max heap after updating. In real case, we often use the simplified version, that is to say, ignore the process of reforming the max heap. Instead, directly search the next largest value in the 'max heap'.
* **Divisive** Top-down (Details refer to [this](https://arxiv.org/pdf/cond-mat/0308217.pdf))

    Remove the edges with the highest betweeness from the graph, and recalculate after each change.

    The betweeness have three commonly-used definition:

    i) Shortest parth betweeness: use BFS algorithm

    ii) Random walk betweeness: How often on average random walks starting at vertex s will pass down a particular edge from vertex v to vertex w before reaching the target t. For more details, can refer to [this.](https://arxiv.org/pdf/cond-mat/0308217.pdf)

    (Probability matrix) M = (Adjacency matrix) A * (Diagonal matrix of the degree of each node) D^(-1)

    To prevent the edge reaching *t*, we delete the t's row and column. Denoted as: *M<sub>t</sub>*

    Assume we start from s, take *n* steps to end up with node v (different with t). Denoted as: *[M<sub>t</sub><sup>n</sup>]<sub>vs</sub>*

    Summing over all n (the mean number of times that a walk of any length) traverses the edge from v to w is:
    *k<sub>v</sub><sup>-1</sup>[I-M<sub>t</sub>]<sub>vs</sub><sup>-1</sup>.* Denotes as *V<sub>v</sub>*.

    Similar for from w to v.

    The random walk betweeness of edge (v,w) is |V<sub>v</sub>-V<sub>w</sub>|.

    iii) Current-flow betweeness

    Disadvantage: Demand heavy computational power, O(m²n) with m edges and n nodes, O(n³) for a spare graph.
* **Louvain algorithm** (Modularity Optimazation)
    It will merge a large network into several communities according to which types of grouping will give the largest increase in Q (formula is give in the orginal paper). Do it repeatedly until get the required number of communities.

    [Original paper.](https://arxiv.org/pdf/0803.0476.pdf)

    [Code implementation here.](https://github.com/taynaud/python-louvain)
* **Markov Chain Algorithm** (Unsupervised)

    MCL is based on: if start traversing from a random node, it is more likely to traverse inside a cluster instead of traverse between clusters.

    Flow is easier within dense regions than across sparse boundaries, however, in the long run this effect disappears.
    This process can be oberserved:

    ![MCL Elaboration](https://github.com/LiMengyang990726/Module-Detection/blob/master/MCL%20Process.png)

    The process is done through several round of raising to a non-negative power, then re-normalizing.

    Raising to a non-negative power is called: Expansion.

    Re-normalizing is called: **MCL Inflation**, and this is in order to strengthens strong currents, and weakens already weak currents. The formula for inflation is shown as below:

    ![MCL Inflation](https://github.com/LiMengyang990726/Module-Detection/blob/master/MCL%20Inflation.png)

    Repeat the above process until it reaches a *convergent* state. **Attractors** will attract the postive values in the same row, and sometimes overlapping clusters will occur.

    Disadvantage: For clusters with *large diameters*, MCL requires long expansion and low inflation, and requires many itereations as small changes are turbulent to the whole. *Speed* could be further improved by: setting small values to zero at beginning

    [Understanding purpose.](https://www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf)

    [Original paper.](https://dspace.library.uu.nl/bitstream/handle/1874/848/full.pdf?sequence=1&isAllowed=y)


    [Implementation code](https://www.micans.org/mcl/)

    [Detailed implementation code](https://github.com/GuyAllard/markov_clustering)

    [Analysis on Protein-Protein networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1637120/)
* **Diamond Algorithm
## Networkx Environment setup
* **Install Networkx**

    Install in terminal: `$ pip install networkx`.

    Details refer to [here.](https://networkx.github.io/documentation/stable/install.html)
* **Operation to the remote server through terminal**

   1) Copy a directory:
   `scp -r <local/file/path> username@IPaddress:<remote path>`

      E.g. `scp -r ~/Workspaces/Module-Detection lime0400@172.21.32.87:~`

      E.g. `scp -r ~/Workspaces/Module-Detection/diamond_self.py lime0400@172.21.32.87:/home2/lime0400/Module-Detection`

      [Source](https://medium.com/dev-blogs/transferring-files-between-remote-server-and-local-system-133d78d58137)
  2) Edit a file inside the remote server:

      `vim <file name>`

      press `i` to enter `insert` mode

      after the change, press  `esc` to exit from the `insert` mode

      key in `:wq` to save the changes

      Open the SSH access in MAC OS(local): `systemsetup -f -setremotelogin o

      Transfering files from remote server to local desktop: `scp -r gene-network.png MCL\ Inflation.png limengyang@172.22.188.255:/Users/limengyang/Module-Detection/
`

      [Source](https://help.dreamhost.com/hc/en-us/articles/115006413028-Creating-and-editing-a-file-via-SSH)
* **Setup Microsoft Remote desktop**

    Intall the version 10

* **File explanation**

    `gene-network.tsv` contains all nodes and edges that are in our experiment (extracted from the `interactome.tsv` file)

    `gene-disease0` contains the seed nodes of 70 diseases after small modification to feed the gene-visualization.py file

    `gene-disease` contains the seed nodes of 70 diseases (original)

## Recommended Structure
-- Module-detection
  |
  |
  --data
    |
    |
    --dataset
      |
      |
      --alldatas (example format is shown in the dataset folder)
  |
  |
  --FunctioinalFeatures
    |
    |
    --DataSetFeatures
      |
      |
      --result during execution will be stored here
    |
    |
    --functionalFeatures.py(all you need to do is to change corresponding paths)
  |
  |
  --SequenceFeatures
    |
    |
    --DataSetFeatures
      |
      |
      --result during execution will be stored here
    |
    |
    --sequenceFeatures.py(all you need to do is to change corresponding paths)
  |
  |
  --TopologicalFeatures
    |
    |
    --DataSetFeatures
      |
      |
      --result during execution will be stored here
    |
    |
    --topologicalFeatures.py(all you need to do is to change corresponding paths)
