
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               null_val: float = 0.0) -> torch.Tensor:
    mask = (target != null_val).float()
    m = mask.mean()
    if m > 0:
        mask = mask / m
    mask = torch.nan_to_num(mask)
    loss = (pred - target).abs() * mask
    return torch.nan_to_num(loss).mean()

def masked_mape(pred: torch.Tensor, target: torch.Tensor,
                null_val: float = 0.0) -> torch.Tensor:
    mask = (target != null_val).float()
    m = mask.mean()
    if m > 0:
        mask = mask / m
    mask = torch.nan_to_num(mask)
    loss = (pred - target).abs() / (target.abs() + 1e-8) * mask
    return torch.nan_to_num(loss).mean()


def masked_rmse(pred: torch.Tensor, target: torch.Tensor,
                null_val: float = 0.0) -> torch.Tensor:
    mask = (target != null_val).float()
    m = mask.mean()
    if m > 0:
        mask = mask / m
    mask = torch.nan_to_num(mask)
    loss = (pred - target) ** 2 * mask
    return torch.sqrt(torch.nan_to_num(loss).mean())

def granger_sparsity_loss(gc: torch.Tensor, lambda_sparse: float) -> torch.Tensor:
    l1 = gc.abs().mean()
    gc_T = gc.transpose(1, 2)
    symm_penalty = F.mse_loss(gc, gc_T.detach())
    return lambda_sparse * l1 - 5e-5 * symm_penalty

def causal_consistency_loss(h_fused: torch.Tensor, h_prior: torch.Tensor,
                            gc_p2j: torch.Tensor, lambda_cons: float) -> torch.Tensor:
    causal_support = gc_p2j.sum(dim=-1)
    cs_max = causal_support.max(dim=1, keepdim=True).values + 1e-8
    causal_support_n = causal_support / cs_max

    delta = (h_fused - h_prior).norm(dim=-1)
    low_support = 1 - causal_support_n
    penalty_low = (low_support * delta).mean()
    min_expected_delta = 0.01
    penalty_high = (causal_support_n * F.relu(min_expected_delta - delta)).mean()
    loss = penalty_low + penalty_high
    return lambda_cons * loss

# Unified Objective Loss

class SCauCELoss(nn.Module):
    def __init__(
        self,
        lambda_sparse: float,
        lambda_cons: float,
        use_cons_loss: bool,
    ):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_cons = lambda_cons
        self.use_cons_loss = use_cons_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gc: torch.Tensor,
        hub_idx: torch.Tensor, 
        h_fused: torch.Tensor,
        h_prior: torch.Tensor,
        gc_p2j: torch.Tensor,
        warmup_w: float = 1.0,
    ):
        if target.dim() == 3 and target.shape != pred.shape:
            if target.shape[1] == pred.shape[2] and target.shape[2] == pred.shape[1]:
                target = target.permute(0, 2, 1)

        pred_loss = masked_mae(pred, target)
        gc_loss = warmup_w * granger_sparsity_loss(gc, self.lambda_sparse)

        cons_loss = torch.tensor(0.0, device=pred.device)
        if self.use_cons_loss:
            cons_loss = warmup_w * causal_consistency_loss(
                h_fused, h_prior, gc_p2j, self.lambda_cons
            )

        total = pred_loss + gc_loss + cons_loss

        info = {
            'total': total.item(),
            'pred': pred_loss.item(),
            'gc': gc_loss.item(),
            'consist': cons_loss.item(),
        }
        return total, info