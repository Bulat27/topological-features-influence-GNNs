import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_networkx
import pickle as pk
import os
import glob
import math

# row-normalizes the values in x (tensor of features) to sum-up to one
def SumToOneNormalization(x):
  return x.div(x.sum(dim=-1, keepdim=True).clamp(min=1.))

# min-max normalization of a tensor across columns
def MinMaxNormalization(x):
  return (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)

# standard normalization of a tensor across columns
def StandardNormalization(x):
  return (x - x.mean(dim=0)) / x.std(dim=0)

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

def retrieve_test_accs_ensemble(dataset_name, gnn_name):
  paths = os.path.join('./results',dataset_name, gnn_name,'*ensemble*')
  best_avg_acc = 0.0
  best_accs = list()
  for path in glob.glob(paths):
    curr_file = load_results(path)
    if curr_file['avg_acc'] > best_avg_acc:
      best_avg_acc = curr_file['avg_acc']
      best_accs = curr_file['test_accs']
  return best_accs

def retrieve_accs(dataset_name, gnn_name, word, split_accs):
  paths = os.path.join('./results',dataset_name, gnn_name,f'*{word}*')
  best_file = None
  best_global_acc = 0.0
  for path in glob.glob(paths):
    curr_file = load_results(path)
    curr_file_avg_acc = 0.0
    for key in curr_file.keys():
      curr_file_avg_acc = curr_file_avg_acc + curr_file[key]['avg_acc']
    curr_file_avg_acc = curr_file_avg_acc / len(curr_file.keys())
    
    if curr_file_avg_acc > best_global_acc:
      best_global_acc = curr_file_avg_acc
      best_file = curr_file
  
  if word == 'base':
    return [best_file['original'][split_accs], best_file['structural'][split_accs], best_file['positional'][split_accs], best_file['original-structural'][split_accs], best_file['original-positional'][split_accs], best_file['structural-positional'][split_accs], best_file['original-structural-positional'][split_accs]]
  else:
    best_avg_acc = 0.0
    best_split_accs = list()
    for key in best_file.keys():
      if best_file[key]['avg_acc'] > best_avg_acc:
        best_avg_acc = best_file[key]['avg_acc']
        best_split_accs = best_file[key][split_accs]
    return best_split_accs
  

def plot_test_val_accs_gnn(dataset_name, gnn_name):
  N_RUNS = 10
  N_EPOCHS = 200

  base_all_models_accs = retrieve_accs(dataset_name, gnn_name, 'base', 'test_accs')
  mlp_accs = retrieve_accs(dataset_name,gnn_name,'pre', 'test_accs')
  # extend ensemble test accs list to size 10 (cause it's 5)
  ensemble_accs = retrieve_test_accs_ensemble(dataset_name,gnn_name)
  ensemble_accs = ensemble_accs * math.ceil(N_RUNS / len(ensemble_accs))  # expand to at least the wanted size
  ensemble_accs = ensemble_accs[:N_RUNS]

  all_models_names = [
    'original', 
    'structural', 
    'positional',
    'original-structural',
    'original-positional',
    'structural-positional',
    'original-structural-positional',
    'best_mlp',
    'best_ensemble']

  #plt.figure(figsize=(5, 10))
  #plt.rcParams.update({'font.size': 14})

  plt.rc('xtick',labelsize=14)
  plt.rc('ytick',labelsize=14)
  # first plot with avg accs over 10 runs
  if dataset_name == 'arxiv':
    plt.figure(figsize=(15, 6))
    _ = plt.subplot(1,2,1)

  for model_accs in base_all_models_accs:
    plt.plot(np.arange(N_RUNS)+1, model_accs, linewidth=2)

  plt.plot(np.arange(N_RUNS)+1, mlp_accs, linewidth=2)
  plt.plot(np.arange(N_RUNS)+1, ensemble_accs, linewidth=2)

  _ = plt.legend(all_models_names, fontsize="12.5", loc ="lower left")
  plt.grid('on'), plt.xlabel('Run',fontsize="18"), plt.ylabel('Test accuracy',fontsize="18")
  plt.title(f'{gnn_name.upper()} test accuracy on {dataset_name} dataset',fontsize="19")
  plt.xticks(fontsize=14)

  if dataset_name == 'arxiv':
    # second plot with val accs over 200 epochs
    base_all_models_accs = retrieve_accs(dataset_name, gnn_name, 'base', 'val_accs')
    mlp_accs = retrieve_accs(dataset_name,gnn_name,'pre', 'val_accs')
    _ = plt.subplot(1,2,2)
    for model_accs in base_all_models_accs:
      plt.plot(np.arange(N_EPOCHS)+1, model_accs, linewidth=1.5)

    plt.plot(np.arange(N_EPOCHS)+1, mlp_accs, linewidth=1.5)
    
    _ = plt.legend(all_models_names, fontsize="12.5", loc ="lower right")
    plt.grid('on'), plt.xlabel('Epoch',fontsize="18"), plt.ylabel('Validation accuracy',fontsize="18")
    plt.title(f'{gnn_name.upper()} validation accuracy on {dataset_name} dataset', fontsize="19")

  plt.savefig(os.path.join('./plots',f'{gnn_name}_{dataset_name}.pdf'))
  #plt.show()



#plot_test_val_accs_gnn('citeseer','gcn')
