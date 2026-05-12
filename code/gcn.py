import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    2-layer GCN — Kipf & Welling, ICLR 2017.

    Uses PyG's GCNConv which handles the renormalisation trick internally.

    Args:
        in_channels     : input feature dimension
        hidden_channels : hidden layer dimension (paper uses 64 for citation nets)
        out_channels    : number of classes
        dropout         : dropout probability (paper uses 0.5)
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
