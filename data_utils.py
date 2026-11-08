
import numpy as np
import torch
import os

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)
    def inverse_transform(self, data):
        return (data * self.std) + self.mean
class SCauCEDataLoader:
    def __init__(self, xs, ys, inds, batch_size, days=288, shuffle=True, te_seqs=None):
        self.xs = xs
        self.ys = ys
        self.inds = inds
        self.te_seqs = te_seqs  # (N, T_in, 2) full temporal sequences
        self.batch_size = batch_size
        self.days = days
        self.shuffle = shuffle
        self.size = len(xs)
        self.num_batch = int(np.ceil(self.size / batch_size))
        self.current_ind = 0
        
    def shuffle_data(self):
        if self.shuffle:
            permutation = np.random.permutation(self.size)
            self.xs = self.xs[permutation]
            self.ys = self.ys[permutation]
            self.inds = self.inds[permutation]
            if self.te_seqs is not None:
                self.te_seqs = self.te_seqs[permutation]
    
    def get_iterator(self):
        self.current_ind = 0
        self.shuffle_data()
        
        def _wrapper():
            while self.current_ind < self.num_batch:
                start = self.current_ind * self.batch_size
                end = min(start + self.batch_size, self.size)
                x_batch = self.xs[start:end]
                y_batch = self.ys[start:end]
                ind_batch = self.inds[start:end] % self.days
                te_batch = self.te_seqs[start:end] if self.te_seqs is not None else None
                
                self.current_ind += 1
                yield (x_batch, y_batch, ind_batch, te_batch)
        
        return _wrapper()

def load_scauce_data(data_path, batch_size, in_seq=12, out_seq=12, days=288):  
    print(f"Loading SCauCE data from: {data_path}")
    data = np.load(data_path)
    
    # Extract data
    train_x = data['train_x']     
    train_y = data['train_target'] 
    val_x = data['val_x']
    val_y = data['val_target']
    test_x = data['test_x']
    test_y = data['test_target']
    # Temporal embeddings 
    train_te = data['trainTE']     
    val_te = data['valTE']
    test_te = data['testTE']

    if len(train_te) > days and np.unique(train_te[:, :, 1]).size < 2:
        raise ValueError(
            "The training time-of-day indices are constant. Regenerate the "
            "samples with real timestamps or an explicit start_datetime."
        )
    for name, te in [('train', train_te), ('val', val_te), ('test', test_te)]:
        if te.shape[-1] != 2 or te[:, :, 0].min() < 0 or te[:, :, 0].max() > 6 \
                or te[:, :, 1].min() < 0 or te[:, :, 1].max() >= days:
            raise ValueError(f"Invalid {name} temporal indices: expected [day 0-6, slot 0-{days-1}].")
    
    num_nodes = train_x.shape[2]

    print(f"Train: x={train_x.shape}, y={train_y.shape}")
    print(f"Val: x={val_x.shape}, y={val_y.shape}")
    print(f"Test: x={test_x.shape}, y={test_y.shape}")
    
    train_x_expanded = np.concatenate([train_x, train_x], axis=-1)
    val_x_expanded = np.concatenate([val_x, val_x], axis=-1)
    test_x_expanded = np.concatenate([test_x, test_x], axis=-1)
    
    scaler = StandardScaler(
        mean=train_x[..., 0].mean(),
        std=train_x[..., 0].std()
    )
    
    # Normalize inputs 
    for si in range(train_x_expanded.shape[-1]):
        mean_feat = train_x_expanded[..., si].mean()
        std_feat = train_x_expanded[..., si].std()
        train_x_expanded[..., si] = (train_x_expanded[..., si] - mean_feat) / (std_feat + 1e-8)
        val_x_expanded[..., si] = (val_x_expanded[..., si] - mean_feat) / (std_feat + 1e-8)
        test_x_expanded[..., si] = (test_x_expanded[..., si] - mean_feat) / (std_feat + 1e-8)
    
   
    train_ind = train_te[:, 0, 1].astype(np.int64)
    val_ind   = val_te[:, 0, 1].astype(np.int64)
    test_ind  = test_te[:, 0, 1].astype(np.int64)
    train_te_in = train_te[:, :in_seq, :].astype(np.int64)
    val_te_in   = val_te[:, :in_seq, :].astype(np.int64)
    test_te_in  = test_te[:, :in_seq, :].astype(np.int64)

    print(f"Tod range in train TE: min={train_te_in[...,1].min()}, max={train_te_in[...,1].max()}")
    print(f"Dow range in train TE: min={train_te_in[...,0].min()}, max={train_te_in[...,0].max()}")

    # data loaders
    train_loader = SCauCEDataLoader(train_x_expanded, train_y, train_ind, batch_size, days=days, shuffle=True,  te_seqs=train_te_in)
    val_loader   = SCauCEDataLoader(val_x_expanded,   val_y,   val_ind,   batch_size, days=days, shuffle=False, te_seqs=val_te_in)
    test_loader  = SCauCEDataLoader(test_x_expanded,  test_y,  test_ind,  batch_size, days=days, shuffle=False, te_seqs=test_te_in)
    
    dataloader = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'y_test': test_y,
        'num_nodes': num_nodes
    }
    
    return dataloader
