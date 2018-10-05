
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
* **Modularity**: The original formula was introducted in this [paper](https://arxiv.org/pdf/cond-mat/0308217.pdf).
              Then there are some modified forms introduced [here](https://en.wikipedia.org/wiki/Modularity_(networks)).
              Another form is introduced as the picture.
* **Centraility**: a significant measurement to find out the most important nodes
               Have three types: Degree centraility, Closeness centraility, and Betweeness centraility

## Module detection algorithm
* **Agglomerative** Bottom-up
    1. Original way: Expressed in the Dendrogram form.
                     Merge pair of clusters that are closest. [Please refer to this for more details.](https://www.youtube.com/watch?v=XJ3194AmH40)
              
        Disadvantage: Tendency to find only the cores of community
    2. Greedy way: Optimize modularity approach.
                   [Please refer to this for more details.](https://arxiv.org/pdf/cond-mat/0408187.pdf)
* **Divisive** Top-down (Details refer to [this](https://arxiv.org/pdf/cond-mat/0308217.pdf))
    Remove the edges with the highest betweeness from the graph, and recalculate after each change.
    The betweeness have three commonly-used definition:
    1. Shortest parth betweeness: use BFS algorithm
    2. Random walk betweeness: 
    3. Current-flow betweeness
* **Louvain algorithm** (Modularity Optimazation)
    [Details elaboration](https://arxiv.org/pdf/0803.0476.pdf) 
    [Code implementation here](https://github.com/taynaud/python-louvain)
* **Markov Chain Algorithm** (Unsupervised)
