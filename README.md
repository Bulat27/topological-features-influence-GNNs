## Evaluation of topological features’ influence on a node classification task using different GNN architectures

The folder `colab notebooks` contains the three Google Colab notebooks we used to run all of the experiments related to this project. In particular, it contains:
* **Experiments_GAT.ipynb**: used to run the experiments for the GAT architecture;
* **Experiments_GCN.ipynb**: used to run the experiments for the GCN architecture;
* **Experiments_linear.ipynb**: used to run the experiments for the linear models Logistic Regression, SVM and Decision Tree.

**Note**: due to stochasticity present in some of the models we employed, results replicated with the notebooks above may have small differences from the ones in the original report.

The folder `models` contains the implementations of the GCN and GAT models in Pytorch Geometric, along with the MLP and ensemble methods used for topological features integration.

The folder `features` contains the code used to compute the structural features using NetworkX, and the positional features using an implementation of Node2Vec in Pytorch Geometric.

The folders `additional` and `plots` contains helper functions (e.g. datasets loading, normalization functions etc.) and plots of the results, respectively.
