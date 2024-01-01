import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_networkx
import pickle as pk
import os

# row-normalizes the values in x (tensor of features) to sum-up to one
def SumToOneNormalization(x):
  return x.div(x.sum(dim=-1, keepdim=True).clamp(min=1.))

# min-max normalization of a tensor across columns
def MinMaxNormalization(x):
  return (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)

# standard normalization of a tensor across columns
def StandardNormalization(x):
  return (x - x.mean(dim=0)) / x.std(dim=0)

def plot_results(n_epochs, train_losses, train_accs, val_losses, val_accs):
  N_EPOCHS = n_epochs
  # Plot results
  plt.figure(figsize=(20, 6))
  _ = plt.subplot(1,2,1)
  plt.plot(np.arange(N_EPOCHS)+1, train_losses, linewidth=3)
  plt.plot(np.arange(N_EPOCHS)+1, val_losses, linewidth=3)
  _ = plt.legend(['Train', 'Validation'])
  plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Loss')

  _ = plt.subplot(1,2,2)
  plt.plot(np.arange(N_EPOCHS)+1, train_accs, linewidth=3)
  plt.plot(np.arange(N_EPOCHS)+1, val_accs, linewidth=3)
  _ = plt.legend(['Train', 'Validation'])
  plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Accuracy')

# concatenate the tensors passed its variable-length input
def concatenate(*tensors):
  return torch.cat(tensors,dim=-1)

# convert a python dictionary to a pytorch tensor of size [len(d),1]
def dict_to_tensor(d):
  return torch.tensor(list(d.values())).reshape(len(d),1)

# convert the data from directed to undirected
def data_to_undirected(data):
  G = compute_graph(data,True)
  data_unidirected = from_networkx(G)

  # restore the attributes
  data_unidirected.name = data.name
  data_unidirected.num_classes = data.num_classes
  data_unidirected.x = data.x
  data_unidirected.y = data.y
  data_unidirected.train_mask = data.train_mask
  data_unidirected.val_mask = data.val_mask
  data_unidirected.test_mask = data.test_mask

  return G, data_unidirected

# compute networkx graph from data
def compute_graph(data, undirected):
  return to_networkx(data, to_undirected=undirected)

# obj is the object to be saved
# obj_name is the name we want to assign to its file
def save_results(obj, obj_name):
  filehandler = open(obj_name, 'wb')
  pk.dump(obj, filehandler)

# filename is the name of the object we want to restore
def load_results(filename):
  filehandler = open(filename, 'rb')
  obj = pk.load(filehandler)
  return obj