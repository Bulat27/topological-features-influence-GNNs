import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_networkx

# z-score normalization of a tensor
def NormalizeTensor(data):
  return (data - torch.mean(data)) / torch.std(data)

# min-max normalization of a dict
def MinMaxNormalization(x):
  x_val = list(x.values())
  x_min = min(x_val)
  x_max = max(x_val)
  for key in x.keys():
    x[key] = float((x[key]-x_min)/(x_max-x_min))
  return x

# z-score normalization of a dict
def ZScoreNormalization(x):
  x_val = list(x.values())
  x_mean = np.mean(x_val)
  x_std = np.std(x_val)
  for key in x.keys():
    x[key] = float((x[key]-x_mean)/x_std)
  return x

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

# concatenate two 2D tensors so to extend the second dimension
# e.g. x.size = [4,2], y.size = [4,5] -> concatenate(x,y).size = [4,7]
def concatenate(tensor1, tensor2):
  return torch.cat((tensor1,tensor2),dim=-1)

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

def compute_graph(data, undirected):
  return to_networkx(data, to_undirected=undirected)