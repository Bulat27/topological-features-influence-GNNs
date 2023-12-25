import torch
import numpy as np
import matplotlib.pyplot as plt

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