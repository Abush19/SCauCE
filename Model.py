
import torch
import torch.nn as nn
import torch.nn.functional as F

# Node-Specific Pe-riodic Encoder (NSPE)

class NSPE(nn.Module):
    def __init__(self, num_nodes: int, in_steps: int,
                 tod_size: int, dow_size: int,
                 embed_dim: int, out_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.tod_embedding = nn.Embedding(tod_size, embed_dim)
        self.dow_embedding = nn.Embedding(dow_size, embed_dim)
        combined = embed_dim * 2
        self.node_gate = nn.Parameter(torch.ones(num_nodes, combined))
        self.node_bias = nn.Parameter(torch.zeros(num_nodes, combined))
        self.proj = nn.Linear(combined, out_dim)

    def forward(self, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        tod_e = self.tod_embedding(tod)
        dow_e = self.dow_embedding(dow)
        global_e = torch.cat([tod_e, dow_e], dim=-1)
        gate = torch.sigmoid(self.node_gate).unsqueeze(0).unsqueeze(2)
        bias = self.node_bias.unsqueeze(0).unsqueeze(2)
        modulated = gate * global_e.unsqueeze(1) + bias
        out = self.proj(modulated)
        return out

# Temporal State Builder (TSB)

class TSB(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 num_lags: int = 3, heads: int = 4):
        super().__init__()
        self.num_lags = num_lags
        dilations = [1, 2, 4][:num_lags]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, kernel_size=3,
                      padding=d, dilation=d)
            for d in dilations
        ])
        self.lag_proj = nn.Linear(out_dim * num_lags, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, C = x.shape
        xr = x.reshape(B * N, T, C).permute(0, 2, 1)
        lag_feats = []
        for conv in self.convs:
            feat = F.relu(conv(xr))
            lag_feats.append(feat.permute(0, 2, 1))
        out = self.lag_proj(torch.cat(lag_feats, dim=-1))
        attn_out, _ = self.attn(out, out, out)
        out = self.norm(out + attn_out)
        out = out.reshape(B, N, T, -1)
        return out

# Causal Compression

class NISPrescreener(nn.Module):
    def __init__(self, node_dim: int, hidden: int, K: int):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, h_seq: torch.Tensor):
        h_last = h_seq[:, :, -1, :]
        h_mean = h_seq.mean(dim=2)
        summary = torch.cat([h_last, h_mean], dim=-1)
        influence = self.mlp(summary).squeeze(-1)
        _, nis_idx = torch.topk(influence, self.K, dim=1)
        return influence, nis_idx

class NeuralGrangerScorer(nn.Module):
    def __init__(self, node_dim: int, hidden: int, num_lags: int):
        super().__init__()
        self.num_lags = num_lags
        self.restrict_enc = nn.Sequential(
            nn.Linear(num_lags * node_dim, hidden),
            nn.ReLU()
        )
        self.unrestrict_enc = nn.Sequential(
            nn.Linear(2 * num_lags * node_dim, hidden),
            nn.ReLU()
        )
        self.pred_head = nn.Linear(hidden, node_dim)
        self.gc_gate = nn.Sequential(
            nn.Linear(2 * num_lags * node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def _score_pairs(self, src_past, tgt_past, tgt_future):
        B, S, L, D = src_past.shape
        T = tgt_past.shape[1]
        src_flat = src_past.reshape(B, S, L * D)
        tgt_flat = tgt_past.reshape(B, T, L * D)
        restr = self.pred_head(self.restrict_enc(tgt_flat))
        src_exp = src_flat.unsqueeze(2).expand(-1, -1, T, -1)
        tgt_exp = tgt_flat.unsqueeze(1).expand(-1, S, -1, -1)
        pair = torch.cat([src_exp, tgt_exp], dim=-1)
        unrestr = self.pred_head(
            self.unrestrict_enc(pair.reshape(B * S * T, -1))
        ).reshape(B, S, T, D)
        tgt_fut_exp = tgt_future.unsqueeze(1).expand(-1, S, -1, -1)
        err_r = (restr.unsqueeze(1) - tgt_fut_exp).pow(2).mean(dim=-1)
        err_u = (unrestr - tgt_fut_exp).pow(2).mean(dim=-1)
        improvement = err_r - err_u
        gate_in = pair.reshape(B * S * T, -1)
        gc_kn = self.gc_gate(gate_in).reshape(B, S, T)
        gc_kn = gc_kn * torch.sigmoid(improvement)
        return gc_kn

    def forward(self, h_seq, nis_idx, direction='out'):
        B, N, T, D = h_seq.shape
        K = nis_idx.shape[1]
        L = self.num_lags
        h_past = h_seq[:, :, -L-1:-1, :]
        h_future = h_seq[:, :, -1, :]
        cand_past = h_past.gather(
            1, nis_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
        )
        gc_full = torch.zeros(B, N, N, device=h_seq.device)
        if direction == 'out':
            gc_kn = self._score_pairs(cand_past, h_past, h_future)
            row_idx = nis_idx.unsqueeze(-1).expand(-1, -1, N)
            gc_full.scatter_(1, row_idx, gc_kn)
        else:
            cand_future = h_future.gather(1, nis_idx.unsqueeze(-1).expand(-1, -1, D))
            gc_nk = self._score_pairs(h_past, cand_past, cand_future)
            col_idx = nis_idx.unsqueeze(1).expand(-1, N, -1)
            gc_full.scatter_(2, col_idx, gc_nk)
        return gc_full

class CausalCompression(nn.Module):
    def __init__(self, node_dim: int, num_nodes: int,
                 K: int, P: int, alpha: float,
                 gc_hidden: int, num_lags: int):
        super().__init__()
        self.K = K
        self.P = P
        self.alpha = alpha
        self.cis = NISPrescreener(node_dim, hidden=gc_hidden // 2, K=K)
        self.gc_scorer = NeuralGrangerScorer(node_dim, gc_hidden, num_lags)

    def forward(self, h_seq):
        B, N, T, D = h_seq.shape
        _, nis_idx = self.cis(h_seq)
        gc_out = self.gc_scorer(h_seq, nis_idx, direction='out')
        gc_in = self.gc_scorer(h_seq, nis_idx, direction='in')
        gc_full = gc_out + gc_in
        h = h_seq[:, :, -1, :]
        inf_k = gc_out.gather(1, nis_idx.unsqueeze(-1).expand(-1, -1, N)).sum(dim=-1)
        in_flow_k = gc_in.gather(2, nis_idx.unsqueeze(1).expand(-1, N, -1)).sum(dim=1)
        hub_balance = torch.minimum(inf_k, in_flow_k)
        inf_k_n = inf_k / (inf_k.max(dim=1, keepdim=True)[0] + 1e-8)
        hub_bal_n = hub_balance / (hub_balance.max(dim=1, keepdim=True)[0] + 1e-8)
        score = self.alpha * inf_k_n + (1 - self.alpha) * hub_bal_n
        _, topP_local = torch.topk(score, self.P, dim=1)
        hub_indices = nis_idx.gather(1, topP_local)
        hub_scores = torch.zeros(B, N, device=h_seq.device)
        hub_scores.scatter_(1, hub_indices, 1.0)
        return hub_indices, gc_full, hub_scores, h

#  Hypergraph Construction

class HypergraphBuilder(nn.Module):
    def __init__(self, node_dim: int, tau: float):
        super().__init__()
        self.tau = tau
        self.mlp_up   = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1))
        self.mlp_down = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1))

    def forward(self, h: torch.Tensor, gc: torch.Tensor,
                hub_idx: torch.Tensor):
        B, N, D = h.shape
        P = hub_idx.shape[1]
        h_hub = h.gather(
            1, hub_idx.unsqueeze(-1).expand(-1, -1, D))
        gc_i2p = gc.gather(
            2, hub_idx.unsqueeze(1).expand(-1, N, -1))
        h_n_exp  = h.unsqueeze(2).expand(-1, -1, P, -1)
        h_hub_exp = h_hub.unsqueeze(1).expand(-1, N, -1, -1)
        ctx_up = self.mlp_up(
            torch.cat([h_n_exp, h_hub_exp], dim=-1)).squeeze(-1)
        H_up_raw = torch.sigmoid(gc_i2p - self.tau) * torch.sigmoid(ctx_up)
        H_up = H_up_raw / (H_up_raw.sum(dim=2, keepdim=True) + 1e-8)
        gc_p2j = gc.gather(
            1, hub_idx.unsqueeze(-1).expand(-1, -1, N)).permute(0, 2, 1)
        ctx_down = self.mlp_down(
            torch.cat([h_n_exp, h_hub_exp], dim=-1)).squeeze(-1)
        H_down_raw = torch.sigmoid(gc_p2j - self.tau) * torch.sigmoid(ctx_down)
        H_down = H_down_raw / (H_down_raw.sum(dim=2, keepdim=True) + 1e-8)
        return H_up, H_down, gc_i2p, gc_p2j

# Causal Expansion
# 1. Converging Phase

class ConvergingPhase(nn.Module):
    def __init__(self, node_dim: int, hub_dim: int):
        super().__init__()
        self.phi = nn.Linear(node_dim, hub_dim)
        self.psi = nn.Linear(node_dim, hub_dim)
        self.alpha_net = nn.Sequential(
            nn.Linear(node_dim * 2, hub_dim),
            nn.ReLU(),
            nn.Linear(hub_dim, hub_dim),
            nn.Sigmoid())
        self.rho = nn.Sequential(
            nn.Linear(hub_dim, hub_dim),
            nn.ReLU())
        self.hub_update = nn.GRUCell(hub_dim, hub_dim)

    def forward(self, h: torch.Tensor, h_hub: torch.Tensor,
                H_up: torch.Tensor, gc_i2p: torch.Tensor,
                tau: float) -> torch.Tensor:
        B, N, D = h.shape
        P = h_hub.shape[1]

        phi_h = self.phi(h).unsqueeze(2).expand(-1, -1, P, -1)
        causal_gate = torch.sigmoid(gc_i2p - tau).unsqueeze(-1)
        h_tilde = phi_h * causal_gate

        h_tilde_w = h_tilde * H_up.unsqueeze(-1)
        sum_phi = h_tilde_w.sum(dim=1)

        psi_h = self.psi(h).unsqueeze(2).expand(-1, -1, P, -1)
        psi_h_w = psi_h * H_up.unsqueeze(-1)
        sum_psi = psi_h_w.sum(dim=1, keepdim=True)
        set_size = H_up.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=2)
        v_minus_i = (sum_psi - psi_h_w) / (set_size - 1)

        h_hub_exp = h_hub.unsqueeze(1).expand(-1, N, -1, -1)
        h_exp     = h.unsqueeze(2).expand(-1, -1, P, -1)
        alpha = self.alpha_net(
            torch.cat([h_exp, h_hub_exp], dim=-1)
        )

        interact = (alpha * psi_h_w * v_minus_i).sum(dim=1)

        z_Up = self.rho(sum_phi + interact)

        hub_dim = z_Up.shape[-1]
        h_hub_proj = h_hub[..., :hub_dim] if h_hub.shape[-1] >= hub_dim \
                     else F.pad(h_hub, (0, hub_dim - h_hub.shape[-1]))
        hub_new = self.hub_update(
            z_Up.reshape(B * P, -1),
            h_hub_proj.reshape(B * P, -1)
        ).reshape(B, P, hub_dim)

        return hub_new

# 2. Diverging Phase

class DivergingPhase(nn.Module):
    def __init__(self, hub_dim: int, node_dim: int):
        super().__init__()
        self.phi_hub  = nn.Linear(hub_dim, node_dim)
        self.beta_net = nn.Sequential(
            nn.Linear(hub_dim + node_dim + 1, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
            nn.Sigmoid()
        )
        self.score_net = nn.Sequential(
            nn.Linear(hub_dim + node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1)
        )
        self.node_gru = nn.GRUCell(node_dim, node_dim)

    def forward(self, hub_new: torch.Tensor, h: torch.Tensor,
                H_down: torch.Tensor, gc_p2j: torch.Tensor,
                tau: float) -> torch.Tensor:
        B, P, hub_dim = hub_new.shape
        N = h.shape[1]
        node_dim = h.shape[-1]

        hub_exp = hub_new.unsqueeze(1).expand(-1, N, -1, -1)
        h_exp   = h.unsqueeze(2).expand(-1, -1, P, -1)
        phi_hub = self.phi_hub(hub_exp)

        # The directed downstream incidence must actually govern propagation.
        omega = H_down.unsqueeze(-1)
        h_tilde_p2j = phi_hub * omega

        beta_in = torch.cat([
            hub_exp, h_exp, gc_p2j.unsqueeze(-1)
        ], dim=-1)
        beta = self.beta_net(beta_in)

        score = self.score_net(torch.cat([hub_exp, h_exp], dim=-1))
        gc_prior = H_down.unsqueeze(-1)
        num = gc_prior * torch.exp(score)
        denom = num.sum(dim=2, keepdim=True) + 1e-8
        alpha_pj = num / denom
        Z_prop = (alpha_pj * beta * h_tilde_p2j).sum(dim=2)

        h_new = self.node_gru(
            Z_prop.reshape(B * N, node_dim),
            h.reshape(B * N, node_dim)
        ).reshape(B, N, node_dim)

        return h_new, Z_prop

# Causal Reliability Calibration (CRC)

class CausalGating(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(node_dim * 2 + 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
            nn.Sigmoid()
        )

    def forward(self, h_prior, h_expanded, gc_p2j):
        c_i = gc_p2j.sum(dim=2, keepdim=True)
        delta_i = (h_expanded - h_prior).norm(dim=-1, keepdim=True)
        gate_in = torch.cat([h_prior, h_expanded, c_i, delta_i], dim=-1)
        gamma = self.gate_net(gate_in)
        return gamma * h_expanded + (1 - gamma) * h_prior

class ForecastingHead(nn.Module):
    def __init__(self, node_dim: int, out_steps: int, 
                 hidden: int, num_layers: int):
        super().__init__()
        layers = []
        in_dim = node_dim * 2
        for i in range(num_layers):
            out_dim = hidden if i < num_layers - 1 else out_steps
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                in_dim = hidden
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_fused: torch.Tensor, z_prop: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([h_fused, z_prop], dim=-1)
        return self.mlp(inp)

# Main SCauCE

class SCauCE(nn.Module):
    def __init__(self, num_nodes: int,in_steps: int,
        out_steps: int, in_dim: int,node_dim: int,hub_dim: int,K: int, P: int,alpha_cbs: float,
        tau: float, tod_size: int, dow_size: int,embed_dim: int, num_lags: int, attn_heads: int, 
        gc_hidden: int,forecast_hidden: int,num_tsb_blocks: int,forecast_hidden_layers: int,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.tau = tau
        self.P = P

        periodic_dim = 32
        self.nspe = NSPE(num_nodes, in_steps, tod_size, dow_size, embed_dim, periodic_dim)
        actual_raw = in_dim * 2  
        self.input_proj = nn.Linear(actual_raw + periodic_dim, node_dim)

        # Stacked TSB blocks with residual connections
        self.tsb_blocks = nn.ModuleList([
            TSB(node_dim, node_dim, num_lags, attn_heads)
            for _ in range(num_tsb_blocks)
        ])
        self.tsb_norm = nn.LayerNorm(node_dim)

        self.compression = CausalCompression(
            node_dim, num_nodes, K, P, alpha_cbs, gc_hidden, num_lags
        )

        self.hub_proj = nn.Linear(node_dim, hub_dim)
        self.hyp_builder = HypergraphBuilder(node_dim, tau)
        self.converging = ConvergingPhase(node_dim, hub_dim)
        self.hub_back_proj = nn.Linear(hub_dim, node_dim)
        self.diverging = DivergingPhase(hub_dim, node_dim)
        self.gating = CausalGating(node_dim)
        self.head = ForecastingHead(node_dim, out_steps, forecast_hidden, forecast_hidden_layers)

    def forward(self, x, tod, dow):
        B, N, T, C = x.shape
        periodic = self.nspe(tod, dow)
        feat = torch.cat([x, periodic], dim=-1)
        feat = self.input_proj(feat)

        # Stacked TSB with residuals
        h_seq = feat
        for block in self.tsb_blocks:
            h_seq = block(h_seq) + h_seq
        h_seq = self.tsb_norm(h_seq)

        hub_idx, gc, hub_scores, h = self.compression(h_seq)

        D = h.shape[-1]
        h_hub_raw = h.gather(1, hub_idx.unsqueeze(-1).expand(-1, -1, D))
        h_hub = self.hub_proj(h_hub_raw)
        H_up, H_down, gc_i2p, gc_p2j = self.hyp_builder(h, gc, hub_idx)
        hub_new = self.converging(h, h_hub, H_up, gc_i2p, self.tau)
        h_expanded, z_prop = self.diverging(hub_new, h, H_down, gc_p2j, self.tau)
        h_fused = self.gating(h, h_expanded, gc_p2j)
        pred = self.head(h_fused, z_prop)
        return pred, gc, hub_scores, hub_idx, h, gc_p2j, h_fused
