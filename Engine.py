
import torch
import torch.nn as nn
import torch.optim as optim
from Model import SCauCE
from Losses import SCauCELoss, masked_mae, masked_mape, masked_rmse

class SCauCETrainer:
    def __init__(self,scaler, num_nodes: int,in_steps: int,out_steps: int,in_dim: int,node_dim: int,hub_dim: int, 
                 K: int,P: int,alpha_cbs: float,tau: float,tod_size: int,dow_size: int,embed_dim: int,num_lags: int,
                 attn_heads: int,gc_hidden: int,forecast_hidden: int,num_tsb_blocks: int,forecast_hidden_layers: int,
                 warmup_epochs: int,lrate: float,wdecay: float,grad_clip: float,use_grad_ckpt: bool,
                 lambda_sparse: float,lambda_cons: float,use_cons_loss: bool,device,
    ):
        self.device = device
        self.scaler = scaler
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip

        # Build model with all params
        self.model = SCauCE( num_nodes=num_nodes,in_steps=in_steps,
            out_steps=out_steps, in_dim=in_dim, node_dim=node_dim,hub_dim=hub_dim,
            K=K, P=P,alpha_cbs=alpha_cbs,tau=tau,tod_size=tod_size,dow_size=dow_size,
            embed_dim=embed_dim,num_lags=num_lags,attn_heads=attn_heads,
            gc_hidden=gc_hidden,forecast_hidden=forecast_hidden,
            num_tsb_blocks=num_tsb_blocks,forecast_hidden_layers=forecast_hidden_layers,
        ).to(self.device)

        if use_grad_ckpt:
            self.model.gradient_checkpointing = True

        self.loss_fn = SCauCELoss(
            lambda_sparse=lambda_sparse,
            # lambda_hub=lambda_hub,
            lambda_cons=lambda_cons,
            # use_hub_loss=use_hub_loss,
            use_cons_loss=use_cons_loss,
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lrate, weight_decay=wdecay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )
        self._lrate = lrate
        self._epoch = 0
        self._h_prior = None
        self._node_dim = node_dim

    def warmup_weight(self, epoch: int) -> float:
        return min(1.0, epoch / max(self.warmup_epochs, 1))

    def _prepare_batch(self, x, y, tod, dow):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        tod = torch.as_tensor(tod, dtype=torch.long, device=self.device)

        if x.dim() == 4 and x.shape[1] != x.shape[2]:
            x = x.permute(0, 2, 1, 3)

        B, N, T, _ = x.shape

        if y.dim() == 4:
            y = y.squeeze(-1)
        if y.dim() == 3 and y.shape[1] != N:
            y = y.permute(0, 2, 1)

        if tod.dim() == 1:
            tod_seq = tod.unsqueeze(1).expand(B, T)
        else:
            tod_seq = tod

        dow_seq = torch.zeros_like(tod_seq)
        return x, y, tod_seq, dow_seq

    def train(self, x, y, tod, epoch: int = 1, te_batch=None):
        self.model.train()
        self.optimizer.zero_grad()
        x, y, tod_seq, dow_seq = self._prepare_batch(x, y, tod, None)
        if te_batch is not None:
            te_t = torch.as_tensor(te_batch, dtype=torch.long, device=self.device)
            dow_seq = te_t[:, :, 0]
            tod_seq = te_t[:, :, 1]
            tod_seq = tod_seq.clamp(0, self.model.nspe.tod_embedding.num_embeddings - 1)
            dow_seq = dow_seq.clamp(0, self.model.nspe.dow_embedding.num_embeddings - 1)
        pred, gc, hub_scores, hub_idx, h_prior, gc_p2j, h_fused = \
            self.model(x, tod_seq, dow_seq)

        y_norm = self.scaler.transform(y)
        warmup = self.warmup_weight(epoch)
        loss, info = self.loss_fn(
            pred, y_norm, gc, hub_idx, h_fused, h_prior, gc_p2j, warmup
        )
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        pred_inv = self.scaler.inverse_transform(pred.detach())
        mae = masked_mae(pred_inv, y).item()
        mape = masked_mape(pred_inv, y).item()
        rmse = masked_rmse(pred_inv, y).item()
        return mae, mape, rmse, info

    @torch.no_grad()
    def eval(self, x, y, tod, te_batch=None):
        self.model.eval()
        x, y, tod_seq, dow_seq = self._prepare_batch(x, y, tod, None)

        if te_batch is not None:
            te_t = torch.as_tensor(te_batch, dtype=torch.long, device=self.device)
            dow_seq = te_t[:, :, 0].clamp(0, self.model.nspe.dow_embedding.num_embeddings - 1)
            tod_seq = te_t[:, :, 1].clamp(0, self.model.nspe.tod_embedding.num_embeddings - 1)

        pred, _, _, _, _, _, _ = self.model(x, tod_seq, dow_seq)
        pred_inv = self.scaler.inverse_transform(pred)
        mae = masked_mae(pred_inv, y).item()
        mape = masked_mape(pred_inv, y).item()
        rmse = masked_rmse(pred_inv, y).item()
        return mae, mape, rmse

    def lr_step(self, epoch: int):
        self.scheduler.step()
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self._lrate * warmup_factor