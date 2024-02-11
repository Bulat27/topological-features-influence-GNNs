## Evaluation of topological features’ influence on a node classification task using different GNN architectures

A detailed description of the task, methodology, and the performed experiments is provided in the report.pdf file.

The folder `colab notebooks` contains the three Google Colab notebooks we used to run all of the experiments related to this project. In particular, it contains:
* **Experiments_GAT.ipynb**: used to run the experiments for the GAT architecture;
* **Experiments_GCN.ipynb**: used to run the experiments for the GCN architecture;
* **Experiments_linear.ipynb**: used to run the experiments for the linear models Logistic Regression, SVM and Decision Tree.

To run a specific .ipynb file on your personal computer, just move it to the root folder and execute its cells.
To run it on Google Colab you can do the following:
1. upload the current repo to Google Drive under `drive/MyDrive/Colab Notebooks`;
2. open the desired .ipynb with Google Colab;
3. load the GDrive partition using the code:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. change the working directory to the repo directory in GDrive with the command:
```bash
%cd drive/MyDrive/Colab\ Notebooks/{repo_name}
```

**Note**: due to stochasticity present in some of the models we employed, results replicated with the notebooks above may have small differences from the ones in the original report.

The folder `models` contains the implementations of the GCN and GAT models in Pytorch Geometric, along with the MLP and ensemble methods used for topological features integration.

The folder `features` contains the code used to compute the structural features using NetworkX, and the positional features using an implementation of Node2Vec in Pytorch Geometric.

The folders `additional` and `plots` contains helper functions (e.g. datasets loading, normalization functions etc.) and plots of the results, respectively.

The folder `results` contains all the experimental results saved using the Python module **pickle**. It is divided into two sub-folders `results/arxiv` and `results/cora`, each containing the results for the respective dataset. These results follow the organization:

* `gat` -> results for the GAT architecture;
* `gcn` -> results for the GCN architecture;
* `linear` -> results for the three linear models.

The folder `results/arxiv/topological features` also includes the structural and positional features extracted from the ArXiv dataset, since they can be quite expensive to compute.
