
import os
import glob
import time
import argparse
import configparser
import numpy as np
import torch
from data_utils import load_scauce_data
from Engine import SCauCETrainer

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def evaluate_per_step(engine, loader, seq_length, scaler, device):
    all_preds, all_reals = [], []

    for x, y, ind, te_batch in loader.get_iterator():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        ind_t = torch.as_tensor(ind, dtype=torch.long, device=device)

        if x_t.dim() == 4 and x_t.shape[1] != x_t.shape[2]:
            x_t = x_t.permute(0, 2, 1, 3)

        B, N, T, _ = x_t.shape

        if te_batch is not None:
            te_t = torch.as_tensor(te_batch, dtype=torch.long, device=device)
            dow_seq = te_t[:, :, 0].clamp(0, 6)
            tod_seq = te_t[:, :, 1].clamp(0, seq_length * 24 - 1)
        else:
            tod_seq = ind_t.unsqueeze(1).expand(B, T)
            dow_seq = torch.zeros_like(tod_seq)
        with torch.no_grad():
            pred, _, _, _, _, _, _ = engine.model(x_t, tod_seq, dow_seq)
        pred_inv = scaler.inverse_transform(pred.cpu().numpy())
        y_np = y_t.cpu().numpy()
        if y_np.ndim == 4:
            y_np = y_np.squeeze(-1)
        if y_np.ndim == 3 and y_np.shape[2] == T:
            y_np = y_np.transpose(0, 2, 1)

        if pred_inv.ndim == 3:
            pred_inv = pred_inv.transpose(0, 2, 1)
        all_preds.append(pred_inv)
        all_reals.append(y_np)
    yhat = np.concatenate(all_preds, axis=0)
    realy = np.concatenate(all_reals, axis=0)

    amae, amape, armse = [], [], []
    for i in range(seq_length):
        pred_i = yhat[:, i, :]
        real_i = realy[:, i, :]
        mae = np.mean(np.abs(pred_i - real_i))
        rmse = np.sqrt(np.mean((pred_i - real_i) ** 2))
        mask = real_i > 0.1
        mape = np.mean(np.abs((pred_i[mask] - real_i[mask]) / real_i[mask])) * 100 \
               if mask.any() else 0.0
        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)

    return np.array(amae), np.array(amape), np.array(armse)

# Main

def main(args):
    device = torch.device(args.device)

    #  Load configuration 
    config = configparser.ConfigParser()
    config.read(args.config)
    data_config = config['Data']
    model_config = config['Model']
    data_path = data_config['graph_signal_matrix_filename']
    num_nodes = int(data_config['num_of_vertices'])
    
    # Get training params
    in_steps = int(model_config['in_steps'])
    out_steps = int(model_config['out_steps'])
    tod_size = int(model_config['tod_size'])
    batch_size = int(model_config['batch_size'])
    epochs = int(model_config['epochs'])
    patience = int(model_config['patience'])
    print_section("Loading Data")
    dataloader = load_scauce_data(
        data_path,
        batch_size,
        in_seq=in_steps,
        out_seq=out_steps,
        days=tod_size,

    )
    scaler = dataloader['scaler']
    print(f"Nodes: {num_nodes} | Train: {dataloader['train_loader'].size} "
          f"| Val: {dataloader['val_loader'].size} "
          f"| Test: {dataloader['test_loader'].size}")
    
    # Build model from config 
    print_section("Building SCauCE Model")
    
    # Build engine params from config - ALL params must be in config
    engine_params = {
        'scaler': scaler,
        'num_nodes': num_nodes,
        'device': device,
        'in_steps': in_steps,
        'out_steps': out_steps,
    }
    
    # Type mapping for config values
    type_map = {
        'in_dim': int,'node_dim': int,'hub_dim': int,'K': int,'P': int,'alpha_cbs': float,'tau': float,
        'tod_size': int,'dow_size': int,'embed_dim': int,'num_lags': int,
        'attn_heads': int,'gc_hidden': int,'forecast_hidden': int,'num_tsb_blocks': int,
        'forecast_hidden_layers': int,'warmup_epochs': int,'lrate': float, 'wdecay': float,
        'grad_clip': float,'use_grad_ckpt': bool,'lambda_sparse': float,
        # 'lambda_hub': float,
        'lambda_cons': float,
        # 'use_hub_loss': bool,
        'use_cons_loss': bool,
    }
    
    for key, typ in type_map.items():
        if key in model_config:
            if typ == bool:
                engine_params[key] = model_config.getboolean(key)
            else:
                engine_params[key] = typ(model_config[key])
        else:
            raise ValueError(f"Missing required config key: {key} in [Model] section")
    
    engine = SCauCETrainer(**engine_params)
    n_params = sum(p.numel() for p in engine.model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    #  Training loop 
    print_section("Training")
    os.makedirs(args.save, exist_ok=True)

    his_loss = []
    train_times = []
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        train_mae, train_mape, train_rmse = [], [], []
        loss_components = {k: [] for k in ['total','pred','gc','consist']}
        t0 = time.time()

        for itera, (x, y, ind, te_batch) in enumerate(
                dataloader['train_loader'].get_iterator()):

            mae, mape, rmse, info = engine.train(x, y, ind, epoch=epoch, te_batch=te_batch)
            train_mae.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            for k in loss_components:
                loss_components[k].append(info[k])

            if itera % args.print_every == 0:
                print(f"  Ep{epoch:03d} Iter{itera:04d} | "
                    f"MAE {mae:.3f}  MAPE {mape*100:.2f}%  RMSE {rmse:.3f}")       

        t_train = time.time() - t0
        train_times.append(t_train)

        valid_mae, valid_mape, valid_rmse = [], [], []
        t1 = time.time()
        for x, y, ind, te_batch in dataloader['val_loader'].get_iterator():
            mae, mape, rmse = engine.eval(x, y, ind, te_batch=te_batch)
            valid_mae.append(mae)
            valid_mape.append(mape)
            valid_rmse.append(rmse)
        t_val = time.time() - t1
        mv_loss = np.mean(valid_mae)
        engine.lr_step(epoch)
        improved = len(his_loss) == 0 or mv_loss < np.min(his_loss)
        if improved:
            patience_cnt = 0
        else:
            patience_cnt += 1
        his_loss.append(mv_loss)

        print(f"\nEpoch {epoch:03d} | "
              f"Train MAE {np.mean(train_mae):.4f}  MAPE {np.mean(train_mape)*100:.2f}%  RMSE {np.mean(train_rmse):.4f} | "
              f"Val MAE {mv_loss:.4f}  MAPE {np.mean(valid_mape)*100:.2f}%  RMSE {np.mean(valid_rmse):.4f} | "
              f"Train {t_train:.1f}s  Val {t_val:.1f}s  "
              f"{'[BEST]' if improved else f'[patience {patience_cnt}/{patience}]'}")

        ckpt_path = os.path.join(args.save, f"epoch_{epoch}_{mv_loss:.4f}.pth")
        torch.save(engine.model.state_dict(), ckpt_path)

        if patience_cnt >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # Final Test Evaluation 
    print_section("Final Test Evaluation")

    best_epoch = np.argmin(his_loss)
    best_val = his_loss[best_epoch]
    best_path = os.path.join(args.save, f"epoch_{best_epoch+1}_{best_val:.4f}.pth")
    if not os.path.exists(best_path):
        files = sorted(glob.glob(os.path.join(args.save, "epoch_*.pth")))
        best_path = files[-1] if files else None

    if best_path and os.path.exists(best_path):
        engine.model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        print(f"Loaded best model: {best_path}")
    else:
        print("Warning: best checkpoint not found; using current weights.")

    # Standard per‑step test metrics 
    amae, amape, armse = evaluate_per_step(
        engine, dataloader['test_loader'],
        out_steps, scaler, device
    )

    print("\nPer-step performance:")
    print(f"{'Step':>6} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
    for i in range(len(amae)):
        print(f"  {i+1:>4}    {amae[i]:8.3f}   {armse[i]:8.3f}   {amape[i]:7.2f}%")

    print_section("Summary")
    print(f"Average MAE  : {np.mean(amae):.4f}")
    print(f"Average MAPE : {np.mean(amape):.2f}%")
    print(f"Average RMSE : {np.mean(armse):.4f}")
    print(f"Total training time: {sum(train_times)/60:.1f} min")

    return amae, amape, armse

# Argument parsing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SCauCE")
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--save', type=str, default='save_models08/SCauCE/',
                        help='Directory to save models')
    parser.add_argument('--print_every', type=int, default=50,
                        help='Print every N iterations')

    args = parser.parse_args()
    main(args)