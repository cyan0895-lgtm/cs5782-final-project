import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class FastGCNConv(nn.Module):
    """Single FastGCN layer with importance-sampled neighbourhood aggregation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index, edge_weight, sample_mask=None):
        N = x.size(0)
        row, col = edge_index   # row = target, col = source

        if sample_mask is not None:
            keep = sample_mask[col]
            row, col, w = row[keep], col[keep], edge_weight[keep]
        else:
            w = edge_weight

        agg = torch.zeros(N, x.size(1), device=x.device)
        agg.index_add_(0, row, w.unsqueeze(-1) * x[col])
        return self.linear(agg)


class FastGCN(nn.Module):
    """
    FastGCN — Chen et al., ICLR 2018.

    Importance-sampled 2-layer GCN.  At training time each layer independently
    samples `sample_size` nodes proportional to ||S_{:,j}||^2 (column L2-norm
    of the normalised adjacency).  Full-graph inference is used at test time.

    Args:
        in_channels     : input feature dimension
        hidden_channels : hidden layer dimension
        out_channels    : number of classes
        sample_size     : nodes sampled per layer per forward pass
                          (paper: 400 for Cora/Citeseer, larger for Pubmed)
        dropout         : dropout probability
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, sample_size: int = 400,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = FastGCNConv(in_channels, hidden_channels)
        self.conv2 = FastGCNConv(hidden_channels, out_channels)
        self.sample_size = sample_size
        self.dropout = dropout

        self._ei: torch.Tensor | None = None
        self._ew: torch.Tensor | None = None
        self._probs: torch.Tensor | None = None

    def reset_parameters(self):
        self.conv1.linear.reset_parameters()
        self.conv2.linear.reset_parameters()
        nn.init.xavier_uniform_(self.conv1.linear.weight)
        nn.init.xavier_uniform_(self.conv2.linear.weight)

    @torch.no_grad()
    def precompute(self, x, edge_index):
        """Precompute normalised adjacency and importance-sampling probabilities."""
        N = x.size(0)
        ei, ew = gcn_norm(edge_index, num_nodes=N,
                          add_self_loops=True, dtype=x.dtype)
        self._ei, self._ew = ei, ew
        # p_j ∝ ||S_{:,j}||^2 = sum of squared weights in column j
        col_norm_sq = torch.zeros(N, device=x.device)
        col_norm_sq.index_add_(0, ei[1], ew ** 2)
        self._probs = col_norm_sq / col_norm_sq.sum()

    def _sample(self, N: int) -> torch.Tensor:
        idx = torch.multinomial(self._probs,
                                num_samples=min(self.sample_size, N),
                                replacement=False)
        mask = torch.zeros(N, dtype=torch.bool, device=idx.device)
        mask[idx] = True
        return mask

    def forward(self, x, edge_index=None):
        ei, ew, N = self._ei, self._ew, x.size(0)
        mask1 = self._sample(N) if self.training else None
        mask2 = self._sample(N) if self.training else None

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, ei, ew, sample_mask=mask1))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, ei, ew, sample_mask=mask2)
