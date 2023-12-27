import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class GCNBase(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCNPre(torch.nn.Module):
    def __init__(self, data, hidden_channels, mlp_hidden_channels):
        super().__init__()

        # Single-layer MLP preprocessing
        self.mlp_linear = nn.Linear(data.num_features, mlp_hidden_channels)

        self.conv1 = GCNConv(mlp_hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        # MLP preprocessing layer

        x = self.mlp_linear(x)
        x = F.relu(x)  

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x        