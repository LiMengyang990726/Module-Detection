
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
* **Divisive** Top-down (Details refer to [this](https://arxiv.org/pdf/cond-mat/0308217.pdf)

    Remove the edges with the highest betweeness from the graph, and recalculate after each change.
    
    The betweeness have three commonly-used definition:

    i) Shortest parth betweeness: use BFS algorithm

    ii) Random walk betweeness: How often on average random walks starting at vertex s will pass down a particular edge from vertex v to vertex w. For more details, can refer to [this](https://arxiv.org/pdf/cond-mat/0308217.pdf) or [this]()

    iii) Current-flow betweeness

    Disadvantage: Demand heavy computational power, O(m²n) with m edges and n nodes, O(n³) for a spare graph.
* **Louvain algorithm** (Modularity Optimazation)
    [Details elaboration](https://arxiv.org/pdf/0803.0476.pdf)
    [Code implementation here](https://github.com/taynaud/python-louvain)
* **Markov Chain Algorithm** (Unsupervised)

## Networkx Environment setup
* **Install Networkx**
* **Setup Microsoft Remote desktop**
