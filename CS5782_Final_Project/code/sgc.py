import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops


class SGC(nn.Module):
    """
    Simple Graph Convolution (SGC)

    Paper idea:
        X_bar = S^K X
        logits = X_bar Theta

    where:
        - S is the normalized adjacency with self-loops
        - K is the propagation order
        - Theta is the only trainable weight matrix

    This implementation supports:
        1. on-the-fly propagation in forward()
        2. optional cached propagation for faster repeated training
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 2,
        bias: bool = True,
        cached: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.dropout = dropout

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        self._cached_x = None
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self._cached_x = None

    def clear_cache(self):
        """Clear cached propagated features."""
        self._cached_x = None

    @torch.no_grad()
    def propagate_features(self, x, edge_index, edge_weight=None):
        """
        Compute X_bar = S^K X using normalized adjacency with self-loops.

        Parameters
        ----------
        x : Tensor [num_nodes, num_features]
            Node feature matrix.
        edge_index : LongTensor [2, num_edges]
            Graph connectivity in COO format.
        edge_weight : Tensor [num_edges], optional
            Edge weights. If None, all edges are treated as weight 1.

        Returns
        -------
        Tensor [num_nodes, num_features]
            Propagated node features.
        """
        num_nodes = x.size(0)

        # gcn_norm does:
        #   A_tilde = A + I
        #   S = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
        edge_index_norm, edge_weight_norm = gcn_norm(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            add_self_loops=True,
            dtype=x.dtype,
        )

        out = x
        row, col = edge_index_norm

        for _ in range(self.K):
            # Sparse propagation:
            # out_i = sum_j S_ij * x_j
            out_next = torch.zeros_like(out)
            out_next.index_add_(0, row, edge_weight_norm.unsqueeze(-1) * out[col])
            out = out_next

        return out

    def precompute(self, x, edge_index, edge_weight=None):
        """
        Precompute and cache S^K X.
        Useful when the graph and node features stay fixed across epochs.
        """
        self._cached_x = self.propagate_features(x, edge_index, edge_weight)
        return self._cached_x

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass:
            1. propagate features K times
            2. apply optional dropout
            3. apply a single linear classifier
        """
        if self.cached and self._cached_x is not None:
            x_prop = self._cached_x
        else:
            x_prop = self.propagate_features(x, edge_index, edge_weight)
            if self.cached:
                self._cached_x = x_prop

        x_prop = F.dropout(x_prop, p=self.dropout, training=self.training)
        logits = self.linear(x_prop)
        return logits


class LogisticRegression(nn.Module):
    """
    Optional plain linear classifier baseline.
    Useful if pipeline precomputes S^K X outside the model.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)