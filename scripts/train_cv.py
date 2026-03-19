import os
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from src.config import (
    DATA_PATHS, SAVE_DIR, DEVICE, HAS_CUDA, SEED,
    BATCH_SIZE, EPOCHS, LR, WD, TARGET_SPECIFICITY,
    USE_GRU, CENTER_WINDOW, LOSS_TYPE, USE_SWA, USE_EMA,
    USE_LAYER_NORM_PRE_RNN, N_FOLDS, NUM_WORKERS
)
from src.data_utils import read_txt_auto
from src.features import compute_channel_stats_for_train
from src.dataset import build_loaders, SeqDataset, _seed_worker
from src.model import CNN_ATT_BiRNN_Model
from src.train_eval import (
    FocalLoss, bce_ls_with_logits, train_one_epoch,
    collect_scores, find_threshold_at_specificity, compute_all_metrics
)

def main():
    print("Using device:", DEVICE, "| CUDA:", HAS_CUDA)
    if HAS_CUDA:
        print("GPU:", torch.cuda.get_device_name(0))

    dfs = []
    for sp, p in DATA_PATHS.items():
        dfs.append(read_txt_auto(p, sp))
    df_all = pd.concat(dfs, ignore_index=True)

    print("Loaded rows:", len(df_all))
    print(df_all["split"].value_counts(dropna=False))

    species_list = sorted(df_all["species"].unique().tolist())
    print("Species list:", species_list)

    summary_rows = []

    for species_name in species_list:
        print(f"\n========== Species: {species_name} (5-fold CV) ==========")
        df_sp = df_all[df_all["species"] == species_name].reset_index(drop=True)
        df_tr = df_sp[df_sp["split"].isin(["train", "val"])].reset_index(drop=True)
        df_te = df_sp[df_sp["split"] == "test"].reset_index(drop=True)

        assert len(df_tr) > 0 and len(df_te) > 0, "Train/Val pool and Test must be non-empty"

        for name, d in [("pool(train+val)", df_tr), ("test", df_te)]:
            p = int((d["label"] == 1).sum())
            n = int((d["label"] == 0).sum())
            print(f"{species_name} {name}: N={len(d)} | pos={p} neg={n} pos_rate={p/(p+n+1e-9):.3f}")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds_out_dir = Path(SAVE_DIR) / f"{species_name}_cv5"
        os.makedirs(folds_out_dir, exist_ok=True)

        fold_results = []
        fold_thresholds = []
        test_fold_preds = []
        test_y_ref = None

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df_tr["seq"].values, df_tr["label"].values), start=1):
            print(f"\n--- Fold {fold_idx}/{N_FOLDS} ---")
            df_fold_tr = df_tr.iloc[tr_idx].reset_index(drop=True)
            df_fold_va = df_tr.iloc[va_idx].reset_index(drop=True)

            mean, std = compute_channel_stats_for_train(df_fold_tr)
            dl_tr, dl_va = build_loaders(df_fold_tr, df_fold_va, BATCH_SIZE, mean, std)

            model = CNN_ATT_BiRNN_Model(
                in_ch=50,
                rnn_type=("gru" if USE_GRU else "lstm"),
                rnn_hidden=256,
                center_window=CENTER_WINDOW,
                use_ln_pre_rnn=USE_LAYER_NORM_PRE_RNN
            ).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=4
            )

            if LOSS_TYPE == "focal":
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
            elif LOSS_TYPE == "label_smooth":
                def criterion(logits, y):
                    return bce_ls_with_logits(logits, y, eps=0.05)
            else:
                pos = max(1, int((df_fold_tr["label"] == 1).sum()))
                neg = max(1, int((df_fold_tr["label"] == 0).sum()))
                pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)

                def criterion(logits, y):
                    return F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)

            swa_model = AveragedModel(model) if USE_SWA else None
            start_swa = int(EPOCHS * 0.7)

            ema_model = CNN_ATT_BiRNN_Model(
                in_ch=50,
                rnn_type=("gru" if USE_GRU else "lstm"),
                rnn_hidden=256,
                center_window=CENTER_WINDOW,
                use_ln_pre_rnn=USE_LAYER_NORM_PRE_RNN
            ).to(DEVICE) if USE_EMA else None

            if ema_model is not None:
                ema_model.load_state_dict(model.state_dict())

            best_auc = -1.0
            best_state = None
            best_epoch = -1
            patience = 8
            bad = 0

            for epoch in range(1, EPOCHS + 1):
                tr_loss = train_one_epoch(
                    model, dl_tr, optimizer, criterion,
                    ema_model=(ema_model if USE_EMA else None), ema_decay=0.999
                )

                va_y, va_scores = collect_scores(model, dl_va)
                try:
                    va_auc = roc_auc_score(va_y, va_scores)
                except Exception:
                    va_auc = float("nan")

                scheduler.step(va_auc)

                if USE_SWA and epoch >= start_swa:
                    swa_model.update_parameters(model)

                improved = va_auc > best_auc
                if improved:
                    best_auc = va_auc
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1

                if epoch % 5 == 0 or epoch == 1:
                    cur_lr = optimizer.param_groups[0]['lr']
                    print(
                        f"Epoch {epoch:03d} | train loss={tr_loss:.5f} | "
                        f"val AUC={va_auc:.4f} | best AUC={best_auc:.4f} @ {best_epoch} | lr={cur_lr:.2e}"
                    )

                if bad >= patience:
                    print(f"Early stopping at epoch {epoch} (no AUC improvement for {patience} epochs).")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            if USE_SWA and swa_model is not None:
                try:
                    model.load_state_dict(swa_model.state_dict())
                    print("Applied SWA weights.")
                except Exception as e:
                    print("SWA load skipped:", e)

            if USE_EMA and ema_model is not None:
                model.load_state_dict(ema_model.state_dict())
                print("Applied EMA weights.")

            va_y, va_scores = collect_scores(model, dl_va)
            thr = find_threshold_at_specificity(va_y, va_scores, TARGET_SPECIFICITY)
            fold_thresholds.append(thr)
            fold_metrics = compute_all_metrics(va_y, scores=va_scores, thr=thr)

            print(
                f"[Fold {fold_idx}] VAL @ t* (Sp≥{TARGET_SPECIFICITY:.2f}) | "
                f"AUC={fold_metrics['AUC']:.4f} | ACC={fold_metrics['ACC']:.4f} | "
                f"Sn={fold_metrics['Sn']:.4f} | Sp={fold_metrics['Sp']:.4f} | "
                f"Pre={fold_metrics['Pre']:.4f} | F1={fold_metrics['F1']:.4f} | thr={thr:.6f}"
            )

            fold_dir = folds_out_dir / f"fold_{fold_idx}"
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(model.state_dict(), str(fold_dir / f"model_{species_name}_fold{fold_idx}.pt"))

            meta = {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "wd": WD,
                "center_window": CENTER_WINDOW,
                "use_gru": USE_GRU,
                "loss_type": LOSS_TYPE,
                "use_swa": USE_SWA,
                "use_ema": USE_EMA,
                "use_ln_pre_rnn": USE_LAYER_NORM_PRE_RNN,
                "thr_star": float(thr),
                "target_specificity": TARGET_SPECIFICITY,
                "env": {
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "cuda": torch.version.cuda if torch.cuda.is_available() else None,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                },
                "fold": fold_idx,
                "n_folds": N_FOLDS,
                "species": species_name,
                "paths": DATA_PATHS
            }

            with open(fold_dir / f"meta_{species_name}_fold{fold_idx}.json", "w") as f:
                json.dump(meta, f, indent=2)

            dl_test = DataLoader(
                SeqDataset(df_te, mean, std),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=_seed_worker,
                persistent_workers=(NUM_WORKERS > 0)
            )

            te_y, te_scores = collect_scores(model, dl_test)
            te_pred = (te_scores >= thr).astype(int)

            if test_y_ref is None:
                test_y_ref = te_y
            else:
                assert np.all(test_y_ref == te_y), "Test labels mismatch across folds"

            np.save(fold_dir / f"test_y_fold{fold_idx}.npy", te_y)
            np.save(fold_dir / f"test_scores_fold{fold_idx}.npy", te_scores)
            np.save(fold_dir / f"test_pred_bin_fold{fold_idx}.npy", te_pred)

            te_metrics = compute_all_metrics(te_y, scores=te_scores, thr=thr)
            print(
                f"[Fold {fold_idx}] TEST @ t* | AUC={te_metrics['AUC']:.4f} | "
                f"ACC={te_metrics['ACC']:.4f} | Sn={te_metrics['Sn']:.4f} | "
                f"Sp={te_metrics['Sp']:.4f} | Pre={te_metrics['Pre']:.4f} | "
                f"F1={te_metrics['F1']:.4f}"
            )

            fold_results.append({
                "fold": fold_idx,
                "VAL_AUC": fold_metrics["AUC"],
                "VAL_ACC": fold_metrics["ACC"],
                "VAL_Sn": fold_metrics["Sn"],
                "VAL_Sp": fold_metrics["Sp"],
                "VAL_Pre": fold_metrics["Pre"],
                "VAL_F1": fold_metrics["F1"],
                "thr": thr,
                "TEST_AUC": te_metrics["AUC"],
                "TEST_ACC": te_metrics["ACC"],
                "TEST_Sn": te_metrics["Sn"],
                "TEST_Sp": te_metrics["Sp"],
                "TEST_Pre": te_metrics["Pre"],
                "TEST_F1": te_metrics["F1"]
            })

            test_fold_preds.append(te_pred)

            del model, dl_tr, dl_va, dl_test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        test_fold_preds = np.stack(test_fold_preds, axis=0)
        votes = test_fold_preds.sum(axis=0)
        majority = int(np.floor(N_FOLDS / 2) + 1)
        y_pred_vote = (votes >= majority).astype(int)

        voting_metrics = compute_all_metrics(test_y_ref, y_pred=y_pred_vote)
        print("\n====== Hard Voting (TEST) ======")
        print(
            f"[{species_name}] TEST HardVote (K={N_FOLDS}) | "
            f"ACC={voting_metrics['ACC']:.4f} | Sn={voting_metrics['Sn']:.4f} | "
            f"Sp={voting_metrics['Sp']:.4f} | Pre={voting_metrics['Pre']:.4f} | "
            f"F1={voting_metrics['F1']:.4f} | TP={voting_metrics['TP']} "
            f"TN={voting_metrics['TN']} FP={voting_metrics['FP']} FN={voting_metrics['FN']}"
        )

        np.save(folds_out_dir / "test_y.npy", test_y_ref)
        np.save(folds_out_dir / "test_pred_hardvote.npy", y_pred_vote)
        np.save(folds_out_dir / "test_votes_count.npy", votes)

        df_fold = pd.DataFrame(fold_results)
        df_fold.to_csv(folds_out_dir / "per_fold_metrics.csv", index=False)

        row = {
            "species": species_name,
            "K": N_FOLDS,
            "HardVote_ACC": voting_metrics["ACC"],
            "HardVote_Sn": voting_metrics["Sn"],
            "HardVote_Sp": voting_metrics["Sp"],
            "HardVote_Pre": voting_metrics["Pre"],
            "HardVote_F1": voting_metrics["F1"],
            "HardVote_TP": voting_metrics["TP"],
            "HardVote_TN": voting_metrics["TN"],
            "HardVote_FP": voting_metrics["FP"],
            "HardVote_FN": voting_metrics["FN"],
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(Path(SAVE_DIR) / "summary_hardvote_per_species.csv", index=False)

    print("\n====== Summary (Hard Voting per species) ======")
    print(df_summary)
    print("Artifacts saved to:", SAVE_DIR)

if __name__ == "__main__":
    main()
