import torch
import numpy as np

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