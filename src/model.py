import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F

# ---------------- model ----------------
class HeteroMLPLayer(nn.Module):
    def __init__(self, in_dims, out_dim):
        super().__init__()

        self.lin = nn.ModuleDict({
            'chemical': nn.Linear(in_dims['chemical'], out_dim),
            'disease':  nn.Linear(in_dims['disease'], out_dim),
        })

    def forward(self, x_dict, edge_index_dict=None):
        return {
            node_type: self.lin[node_type](x)
            for node_type, x in x_dict.items()
        }

class HeteroGraphMLP(nn.Module):
    def __init__(
        self,
        in_channels_dict,
        hidden_channels,
        num_layers,
        dropout,
        MLP_num_layers,
        MLP_dropout,
    ):
        super().__init__()

        self.hidden = hidden_channels
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout)

        # ===== Input projections  =====
        self.lin_dict = nn.ModuleDict({
            'chemical': nn.Linear(in_channels_dict['chemical'], hidden_channels),
            'disease':  nn.Linear(in_channels_dict['disease'], hidden_channels),
        })

        # ===== Automatic shrinking dimensions =====
        dims = [hidden_channels]
        for i in range(1, num_layers + 1):
            dims.append(max(int(hidden_channels * (0.5 ** i)), 1))
        final_emb_dim = dims[-1]

        # ===== Build per-layer input dims =====
        chemical_in_dims = [dims[0]] + dims[1:num_layers+1]
        disease_in_dims  = [dims[0]] + dims[1:num_layers+1]

        # ===== Encoder layers =====
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dims = {
                'chemical': chemical_in_dims[i],
                'disease':  disease_in_dims[i],
            }
            out_dim = dims[i + 1]

            self.mlps.append(HeteroMLPLayer(in_dims, out_dim))
            self.norms.append(nn.ModuleDict({
                'chemical': nn.BatchNorm1d(out_dim),
                'disease':  nn.BatchNorm1d(out_dim),
            }))

        # ===== Output projection =====
        self.post_lin = nn.ModuleDict({
            n: nn.Linear(final_emb_dim, final_emb_dim)
            for n in in_channels_dict
        })

        # ===== Link prediction MLP =====
        self.link_mlp = self._build_mlp(
            input_dim=final_emb_dim * 2,
            num_layers=MLP_num_layers,
            dropout=MLP_dropout,
        )

    # --------------------------------------------------
    def _build_mlp(self, input_dim, num_layers, dropout):
        layers = []
        hidden_dim = max(1, input_dim // 2)

        for _ in range(num_layers):
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
            hidden_dim = max(1, hidden_dim // 2)

        layers.extend([
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        ])
        return nn.Sequential(*layers)

    # --------------------------------------------------
    def forward(self, x_dict, edge_index_dict=None, device=None):
        if device is not None:
            x_dict = {k: v.to(device) for k, v in x_dict.items()}

        # ===== Input projection =====
        h = {
            n: F.relu(self.lin_dict[n](x))
            for n, x in x_dict.items()
        }
        h = {n: self.dropout_layer(h[n]) for n in h}

        # ===== MLP encoder stack =====
        for i, mlp in enumerate(self.mlps):
            h = mlp(h, edge_index_dict)
            for n in h:
                h[n] = self.norms[i][n](h[n])
                h[n] = F.relu(h[n])
                h[n] = self.dropout_layer(h[n])

        # ===== Output projection =====
        for n in h:
            h[n] = F.relu(self.post_lin[n](h[n]))

        return h

    # --------------------------------------------------
    def decode_links(self, src_emb, dst_emb, device=None):
        if device is not None:
            src_emb = src_emb.to(device)
            dst_emb = dst_emb.to(device)
        return self.link_mlp(torch.cat([src_emb, dst_emb], dim=1))

