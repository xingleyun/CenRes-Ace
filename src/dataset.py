import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import SEED, NUM_WORKERS
from .features import make_features_for_seq, CONT_IDX

class SeqDataset(Dataset):
    def __init__(self, df, mean=None, std=None):
        self.df = df.reset_index(drop=True).copy()
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        X = make_features_for_seq(r["seq"])
        if self.mean is not None:
            X[:, CONT_IDX] = (X[:, CONT_IDX] - self.mean[CONT_IDX]) / (self.std[CONT_IDX] + 1e-6)
        y = int(r["label"])
        return torch.from_numpy(X).float(), torch.tensor(y, dtype=torch.long)

def _seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_loaders(df_tr, df_va, batch_size, mean, std):
    g = torch.Generator()
    g.manual_seed(SEED)

    pin = torch.cuda.is_available()
    persistent_workers = (NUM_WORKERS > 0)

    tr = DataLoader(
        SeqDataset(df_tr, mean, std),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=persistent_workers
    )

    va = DataLoader(
        SeqDataset(df_va, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        worker_init_fn=_seed_worker,
        persistent_workers=persistent_workers
    )
    return tr, va
