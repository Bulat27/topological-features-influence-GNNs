import torch
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F

# The number of hidden_channels is multiplied by the number of heads.
class GATBase(torch.nn.Module):
    def __init__(self, data, hidden_channels, heads):
        super().__init__()
        # torch.manual_seed(42)
        self.conv1 = GATConv(in_channels=data.num_features, out_channels=hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(in_channels=hidden_channels*heads, out_channels=data.num_classes, heads=1, dropout=0.6)


    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class GATPre(torch.nn.Module):
    def __init__(self, data, hidden_channels, heads, mlp_hidden_channels):
        super().__init__()

        # Single-layer MLP preprocessing
        self.mlp_linear = nn.Linear(data.num_features, mlp_hidden_channels)

        # GAT layers
        self.conv1 = GATConv(in_channels=mlp_hidden_channels, out_channels=hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(in_channels=hidden_channels*heads, out_channels=data.num_classes, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        # MLP preprocessing layer
        x = self.mlp_linear(x)
        x = F.relu(x)  

        # GAT layers
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x    