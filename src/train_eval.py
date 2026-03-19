import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

from .config import DEVICE

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        loss = (self.alpha * (1 - pt) ** self.gamma) * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def bce_ls_with_logits(logits, targets, eps=0.05):
    targets = targets.float()
    targets = targets * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(logits, targets)

def train_one_epoch(model, loader, optimizer, criterion, ema_model=None, ema_decay=0.999):
    model.train()
    total = 0.0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(x), y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if ema_model is not None:
            with torch.no_grad():
                for p, q in zip(model.parameters(), ema_model.parameters()):
                    q.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        total += loss.item() * y.size(0)

    return total / len(loader.dataset)

@torch.no_grad()
def collect_scores(model, loader):
    model.eval()
    scores = []
    labels = []

    for x, y in loader:
        x = x.to(DEVICE)
        prob = torch.sigmoid(model(x)).detach().cpu().numpy().ravel()
        scores.append(prob)
        labels.append(y.numpy().ravel())

    return np.concatenate(labels).astype(int), np.concatenate(scores)

def find_threshold_at_specificity(y_true, scores, target_sp=0.90):
    fpr, tpr, thr = roc_curve(y_true, scores)
    sp = 1.0 - fpr
    mask = ~np.isinf(thr)
    thr2, sp2 = thr[mask], sp[mask]
    idx = np.where(sp2 >= target_sp)[0]
    if idx.size:
        return float(thr2[idx[-1]])
    neg = scores[np.array(y_true) == 0]
    return float(np.quantile(neg, target_sp)) if neg.size else 0.5

def compute_all_metrics(y_true, scores=None, thr=None, y_pred=None):
    if y_pred is None:
        y_pred = (scores >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sp = tn / (tn + fp + 1e-12)
    sn = tp / (tp + fn + 1e-12)
    pre = tp / (tp + fp + 1e-12)
    f1 = 2 * pre * sn / (pre + sn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    try:
        auc = roc_auc_score(y_true, scores) if scores is not None else float("nan")
    except Exception:
        auc = float("nan")

    return {
        "AUC": float(auc),
        "ACC": float(acc),
        "Sn": float(sn),
        "Sp": float(sp),
        "Pre": float(pre),
        "F1": float(f1),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

def save_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
