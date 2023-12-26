import torch
from torch_geometric.nn import GATConv
import torch.nn as nn

# The number of hidden_channels is multiplied by the number of heads.
class GAT(torch.nn.Module):
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