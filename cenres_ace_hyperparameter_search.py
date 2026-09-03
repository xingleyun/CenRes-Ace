#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CenRes-Ace Stage-1 hyperparameter screening
===========================================

Purpose
-------
Single-model random hyperparameter screening for one species.
This is Stage 1 only:

    TRAIN -> train one full CenRes-Ace model
    VALID -> select best epoch and rank hyperparameters
    TEST  -> report reference metrics for every trial

The full architecture is fixed:
- One-hot + Dnorm + BLOSUM62 + PCP-8, 31 x 50 input
- 1x1 convolutional stem
- 3 multi-scale residual CNN blocks
- channel attention + spatial attention
- BiGRU
- Gaussian center-neighborhood pooling
- MLP classifier

Searchable parameters
---------------------
- batch size
- learning rate
- weight decay
- BiGRU hidden size
- center-neighborhood window
- stem / block / MLP dropout
- MLP hidden size
- label smoothing
- EMA decay
- early-stopping patience

Selection rules
---------------
1. Each trial selects its best epoch primarily by validation AUROC.
2. Validation AUPRC is used as the tie-breaker.
3. Overall hyperparameter ranking uses validation data only.
4. Sn, Sp, ACC, Precision, F1 and MCC are secondary validation metrics
   calculated at a threshold satisfying Sp >= 0.90.
5. Every validation-selected checkpoint is additionally evaluated on TEST.
6. The main test_* columns reproduce the manuscript fixed-specificity convention: the threshold
   is derived on TEST itself by interpolating to Sp=0.90.
7. Additional test_valthr_* columns apply the validation-derived threshold
   to TEST, making the two evaluation conventions directly comparable.
8. Hyperparameter selection still uses validation AUROC only; TEST rankings
   are exploratory views and do not determine best_overall_by_val_AUC.pt.

Resume behavior
---------------
- search_plan.json freezes the randomly sampled trials;
- every epoch saves last_checkpoint.pt;
- interrupted trials resume from the latest completed epoch;
- completed trials are skipped automatically;
- summary tables are rebuilt after every trial.

Runtime configuration
---------------------
The public version uses environment variables rather than machine-specific
absolute paths:

- CENRES_SPECIES: one of the nine supported species names
- CENRES_DATA_ROOT: root directory containing one subdirectory per species
- CENRES_OUTPUT_ROOT: directory for search outputs
- CENRES_GPU_ID: physical GPU index
- CENRES_MAX_TRIALS: optional override of the species-specific search budget

Nine-species use
----------------
The same script is used for all nine species. Species-specific search spaces
are included below; choose the target species with CENRES_SPECIES.
"""

# ==========================================================
# 0) Environment variables: set before importing torch
# ==========================================================
import os

# Physical GPU index used by this process.
GPU_ID = int(os.environ.get("CENRES_GPU_ID", "0"))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

# ==========================================================
# 1) Imports
# ==========================================================
import sys
import re
import gc
import json
import math
import time
import random
import shutil
import hashlib
import traceback
from pathlib import Path
from datetime import datetime, timezone
from collections import OrderedDict

# Public-repository path configuration.
DATA_ROOT = Path(os.environ.get("CENRES_DATA_ROOT", "./data")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("CENRES_OUTPUT_ROOT", "./outputs")).expanduser().resolve()

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    raise RuntimeError(
        "PyTorch could not be imported in the current Python environment. "
        "Please confirm that the intended Python/conda environment is active "
        "and that its PyTorch/CUDA installation is complete. "
        f"Original error: {exc!r}"
    ) from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)


# ==========================================================
# 2) USER CONFIGURATION: select one of nine species
# ==========================================================
# Select one species with the CENRES_SPECIES environment variable.
SELECTED_SPECIES = os.environ.get("CENRES_SPECIES", "Rattus_norvegicus")

# The main test_* columns use the manuscript fixed-specificity evaluation convention:
# derive a threshold from TEST itself by ROC interpolation at Sp=0.90.
OLD_TEST_THRESHOLD_MODE = "original_interp"

# Additional test_valthr_* columns use the threshold derived on VALID.
REPORT_VALIDATION_THRESHOLD_ON_TEST = True

# Search/reproducibility settings shared by all species.
SEARCH_MODE = "random"
INCLUDE_CURRENT_BASELINE = True
SEARCH_SEED = 20260730
TRAIN_SEED = 42

SEQ_LEN = 31
INPUT_DIM = 50
TARGET_SPECIFICITY = 0.90
NUM_WORKERS = 0
USE_AMP = True
PRINT_EVERY = 5
KEEP_LAST_CHECKPOINT_AFTER_FINISH = False
FORCE_RERUN_COMPLETED = False
ECE_N_BINS = 10

# Full-model structure is fixed during Stage 1.
FIXED_MODEL_CONFIG = {
    "cnn_depth": 3,
    "channel_attention": True,
    "spatial_attention": True,
    "se_reduction": 8,
    "spatial_kernel": 7,
    "use_layer_norm_pre_rnn": True,
    "rnn_type": "gru",
    "scheduler": "cosine",
    "eta_min": 1e-5,
}

SPECIES_PRESETS = {
    "Arabidopsis_thaliana": {
        "display_species": "A. thaliana",
        "data_dir": DATA_ROOT / "Arabidopsis_thaliana",
        "max_trials": 200,
        "max_epochs": 100,
        "max_pos_weight": 3.0,
        "current_baseline": {
            "batch_size": 64,
            "lr": 4e-4,
            "weight_decay": 6e-4,
            "rnn_hidden": 224,
            "center_window": 7,
            "stem_dropout": 0.08,
            "block_dropout": 0.08,
            "mlp_hidden": 192,
            "mlp_dropout": 0.45,
            "label_smoothing": 0.03,
            "ema_decay": 0.995,
            "early_stop_patience": 10,
        },
        "hparam_space": OrderedDict({
            "batch_size": [16, 32, 64],
            "lr": [1e-4, 2e-4, 4e-4, 6e-4, 8e-4],
            "weight_decay": [1e-4, 3e-4, 6e-4, 1e-3],
            "rnn_hidden": [128, 192, 224, 256],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.05, 0.08],
            "block_dropout": [0.05, 0.08, 0.12],
            "mlp_hidden": [128, 192, 256],
            "mlp_dropout": [0.30, 0.45, 0.55],
            "label_smoothing": [0.00, 0.03, 0.05],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [8, 12, 16],
        }),
    },
    "Oryza_sativa": {
        "display_species": "O. sativa",
        "data_dir": DATA_ROOT / "Oryza_sativa",
        "max_trials": 200,
        "max_epochs": 100,
        "max_pos_weight": None,
        "current_baseline": {
            "batch_size": 64, "lr": 3e-4, "weight_decay": 5e-4,
            "rnn_hidden": 256, "center_window": 5,
            "stem_dropout": 0.00, "block_dropout": 0.00,
            "mlp_hidden": 256, "mlp_dropout": 0.30,
            "label_smoothing": 0.00, "ema_decay": 0.999,
            "early_stop_patience": 8,
        },
        "hparam_space": OrderedDict({
            "batch_size": [16, 32, 64],
            "lr": [1e-4, 2e-4, 3e-4, 5e-4, 8e-4],
            "weight_decay": [1e-4, 3e-4, 5e-4, 1e-3],
            "rnn_hidden": [128, 192, 224, 256],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.05, 0.08],
            "block_dropout": [0.00, 0.05, 0.08, 0.12],
            "mlp_hidden": [128, 192, 256],
            "mlp_dropout": [0.25, 0.35, 0.45, 0.55],
            "label_smoothing": [0.00, 0.03, 0.05],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [8, 12, 16],
        }),
    },
    "Schistosoma_japonicum": {
        "display_species": "S. japonicum",
        "data_dir": DATA_ROOT / "Schistosoma_japonicum",
        "max_trials": 80,
        "max_epochs": 80,
        "max_pos_weight": None,
        "current_baseline": {
            "batch_size": 64, "lr": 3e-4, "weight_decay": 5e-4,
            "rnn_hidden": 256, "center_window": 5,
            "stem_dropout": 0.00, "block_dropout": 0.00,
            "mlp_hidden": 256, "mlp_dropout": 0.30,
            "label_smoothing": 0.00, "ema_decay": 0.999,
            "early_stop_patience": 8,
        },
        "hparam_space": OrderedDict({
            "batch_size": [16, 32, 64],
            "lr": [1e-4, 2e-4, 3e-4, 5e-4],
            "weight_decay": [1e-4, 3e-4, 5e-4, 1e-3],
            "rnn_hidden": [192, 224, 256, 320],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.03, 0.05, 0.08],
            "block_dropout": [0.00, 0.05, 0.08, 0.12],
            "mlp_hidden": [192, 256, 320],
            "mlp_dropout": [0.25, 0.30, 0.40, 0.50],
            "label_smoothing": [0.00, 0.02, 0.03, 0.05],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [6, 8, 12],
        }),
    },
    "Bacillus_velezensis": {
        "display_species": "B. velezensis",
        "data_dir": DATA_ROOT / "Bacillus_velezensis",
        "max_trials": 70,
        "max_epochs": 70,
        "max_pos_weight": None,
        "current_baseline": {
            "batch_size": 64, "lr": 3e-4, "weight_decay": 5e-4,
            "rnn_hidden": 256, "center_window": 5,
            "stem_dropout": 0.00, "block_dropout": 0.00,
            "mlp_hidden": 256, "mlp_dropout": 0.30,
            "label_smoothing": 0.00, "ema_decay": 0.999,
            "early_stop_patience": 8,
        },
        "hparam_space": OrderedDict({
            "batch_size": [32, 64, 128],
            "lr": [1e-4, 2e-4, 3e-4, 5e-4],
            "weight_decay": [1e-4, 3e-4, 5e-4, 1e-3],
            "rnn_hidden": [192, 256, 320],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.03, 0.05, 0.08],
            "block_dropout": [0.00, 0.05, 0.08, 0.12],
            "mlp_hidden": [192, 256, 320],
            "mlp_dropout": [0.25, 0.30, 0.40, 0.50],
            "label_smoothing": [0.00, 0.02, 0.03, 0.05],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [6, 8, 12],
        }),
    },
    "Plasmodium_falciparum": {
        "display_species": "P. falciparum",
        "data_dir": DATA_ROOT / "Plasmodium_falciparum",
        "max_trials": 200,
        "max_epochs": 70,
        "max_pos_weight": None,
        "current_baseline": {
            "batch_size": 64, "lr": 3e-4, "weight_decay": 5e-4,
            "rnn_hidden": 256, "center_window": 5,
            "stem_dropout": 0.00, "block_dropout": 0.00,
            "mlp_hidden": 256, "mlp_dropout": 0.30,
            "label_smoothing": 0.00, "ema_decay": 0.999,
            "early_stop_patience": 8,
        },
        "hparam_space": OrderedDict({
            "batch_size": [32, 64, 128],
            "lr": [1e-4, 2e-4, 3e-4, 5e-4],
            "weight_decay": [1e-4, 3e-4, 5e-4, 1e-3],
            "rnn_hidden": [192, 256, 320],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.03, 0.05, 0.08],
            "block_dropout": [0.00, 0.05, 0.08, 0.12],
            "mlp_hidden": [192, 256, 320],
            "mlp_dropout": [0.25, 0.30, 0.40, 0.50],
            "label_smoothing": [0.00, 0.02, 0.03, 0.05],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [6, 8, 12],
        }),
    },
    "Escherichia_coli": {
        "display_species": "E. coli",
        "data_dir": DATA_ROOT / "Escherichia_coli",
        "max_trials": 60,
        "max_epochs": 60,
        "max_pos_weight": None,
        "current_baseline": {
            "batch_size": 64, "lr": 3e-4, "weight_decay": 5e-4,
            "rnn_hidden": 256, "center_window": 5,
            "stem_dropout": 0.00, "block_dropout": 0.00,
            "mlp_hidden": 256, "mlp_dropout": 0.30,
            "label_smoothing": 0.00, "ema_decay": 0.999,
            "early_stop_patience": 8,
        },
        "hparam_space": OrderedDict({
            "batch_size": [32, 64, 128],
            "lr": [1e-4, 2e-4, 3e-4, 5e-4],
            "weight_decay": [1e-4, 3e-4, 5e-4, 1e-3],
            "rnn_hidden": [192, 256, 320],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.03, 0.05],
            "block_dropout": [0.00, 0.03, 0.05, 0.08],
            "mlp_hidden": [192, 256, 320],
            "mlp_dropout": [0.20, 0.30, 0.40],
            "label_smoothing": [0.00, 0.02, 0.03],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [5, 8, 10],
        }),
    },
    "Mus_musculus": {
        "display_species": "M. musculus",
        "data_dir": DATA_ROOT / "Mus_musculus",
        "max_trials": 60,
        "max_epochs": 60,
        "max_pos_weight": None,
        "current_baseline": {
            "batch_size": 64, "lr": 3e-4, "weight_decay": 5e-4,
            "rnn_hidden": 256, "center_window": 5,
            "stem_dropout": 0.00, "block_dropout": 0.00,
            "mlp_hidden": 256, "mlp_dropout": 0.30,
            "label_smoothing": 0.00, "ema_decay": 0.999,
            "early_stop_patience": 8,
        },
        "hparam_space": OrderedDict({
            "batch_size": [32, 64, 128],
            "lr": [1e-4, 2e-4, 3e-4, 5e-4],
            "weight_decay": [1e-4, 3e-4, 5e-4, 1e-3],
            "rnn_hidden": [192, 256, 320],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.03, 0.05],
            "block_dropout": [0.00, 0.03, 0.05, 0.08],
            "mlp_hidden": [192, 256, 320],
            "mlp_dropout": [0.20, 0.30, 0.40],
            "label_smoothing": [0.00, 0.02, 0.03],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [5, 8, 10],
        }),
    },
    "Rattus_norvegicus": {
        "display_species": "R. norvegicus",
        "data_dir": DATA_ROOT / "Rattus_norvegicus",
        "max_trials": 60,
        "max_epochs": 60,
        "max_pos_weight": None,
        "current_baseline": {
            "batch_size": 64, "lr": 3e-4, "weight_decay": 5e-4,
            "rnn_hidden": 256, "center_window": 5,
            "stem_dropout": 0.00, "block_dropout": 0.00,
            "mlp_hidden": 256, "mlp_dropout": 0.30,
            "label_smoothing": 0.00, "ema_decay": 0.999,
            "early_stop_patience": 8,
        },
        "hparam_space": OrderedDict({
            "batch_size": [32, 64, 128],
            "lr": [1e-4, 2e-4, 3e-4, 5e-4],
            "weight_decay": [1e-4, 3e-4, 5e-4, 1e-3],
            "rnn_hidden": [192, 256, 320],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.03, 0.05],
            "block_dropout": [0.00, 0.03, 0.05, 0.08],
            "mlp_hidden": [192, 256, 320],
            "mlp_dropout": [0.20, 0.30, 0.40],
            "label_smoothing": [0.00, 0.02, 0.03],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [5, 8, 10],
        }),
    },
    "Saccharomyces_cerevisiae": {
        "display_species": "S. cerevisiae",
        "data_dir": DATA_ROOT / "Saccharomyces_cerevisiae",
        "max_trials": 60,
        "max_epochs": 50,
        "max_pos_weight": None,
        "current_baseline": {
            "batch_size": 64, "lr": 3e-4, "weight_decay": 5e-4,
            "rnn_hidden": 256, "center_window": 5,
            "stem_dropout": 0.00, "block_dropout": 0.00,
            "mlp_hidden": 256, "mlp_dropout": 0.30,
            "label_smoothing": 0.00, "ema_decay": 0.999,
            "early_stop_patience": 8,
        },
        "hparam_space": OrderedDict({
            "batch_size": [64, 128],
            "lr": [1e-4, 2e-4, 3e-4, 5e-4],
            "weight_decay": [1e-4, 3e-4, 5e-4, 1e-3],
            "rnn_hidden": [192, 256, 320],
            "center_window": [3, 5, 7],
            "stem_dropout": [0.00, 0.03, 0.05],
            "block_dropout": [0.00, 0.03, 0.05, 0.08],
            "mlp_hidden": [192, 256, 320],
            "mlp_dropout": [0.20, 0.30, 0.40],
            "label_smoothing": [0.00, 0.02, 0.03],
            "ema_decay": [0.990, 0.995, 0.999],
            "early_stop_patience": [5, 8, 10],
        }),
    },
}

if SELECTED_SPECIES not in SPECIES_PRESETS:
    raise KeyError(
        f"Unknown SELECTED_SPECIES={SELECTED_SPECIES!r}. "
        f"Available: {list(SPECIES_PRESETS)}"
    )

_PRESET = SPECIES_PRESETS[SELECTED_SPECIES]
RAW_SPECIES = SELECTED_SPECIES
DISPLAY_SPECIES = _PRESET["display_species"]
DATA_DIR = Path(_PRESET["data_dir"])
MAX_TRIALS = int(os.environ.get("CENRES_MAX_TRIALS", _PRESET["max_trials"]))
MAX_EPOCHS = int(_PRESET["max_epochs"])
CURRENT_BASELINE = dict(_PRESET["current_baseline"])
HPARAM_SPACE = _PRESET["hparam_space"]
FIXED_MODEL_CONFIG["max_pos_weight"] = _PRESET["max_pos_weight"]

# Every species has an independent output directory.
WORK_ROOT = OUTPUT_ROOT / f"CenResAce_stage1_search_{RAW_SPECIES}"
TRIALS_DIR = WORK_ROOT / "trials"
RESULTS_DIR = WORK_ROOT / "results"
CACHE_DIR = WORK_ROOT / "feature_cache"

# ==========================================================
# 3) Local resume behavior
#
# Persistent local storage is supported; no manual resume import is needed:
# - search_plan.json keeps the sampled trial list fixed;
# - completed trials are skipped automatically;
# - unfinished trials resume from last_checkpoint.pt.
# ==========================================================


# ==========================================================
# 4) Directory creation and reproducibility
# ==========================================================
for directory in [WORK_ROOT, TRIALS_DIR, RESULTS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(TRAIN_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as error:
        print("Warning: deterministic algorithms not fully available:", error)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = bool(USE_AMP and torch.cuda.is_available())

print("=" * 88)
print("CenRes-Ace Stage-1 single-model hyperparameter screening")
print("Species:", DISPLAY_SPECIES, "| raw name:", RAW_SPECIES)
print("Configured GPU ID:", GPU_ID)
print("Device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())
print("AMP enabled:", AMP_ENABLED)
print("Torch version:", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
print("Data directory:", DATA_DIR)
print("Work root:", WORK_ROOT)
print("TEST metrics are reported for every validation-selected checkpoint.")
print("Main test_* convention:", OLD_TEST_THRESHOLD_MODE, "threshold derived on TEST at Sp=0.90")
print("Additional test_valthr_* convention: validation threshold applied to TEST")
print("=" * 88)


# ==========================================================
# 5) Local data-path discovery
# ==========================================================
def find_local_file(filename: str) -> Path:
    """Find one exact file inside DATA_DIR, allowing nested subdirectories."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {DATA_DIR}"
        )

    direct_path = DATA_DIR / filename
    if direct_path.exists():
        return direct_path

    matches = sorted(
        [path for path in DATA_DIR.rglob(filename) if path.is_file()],
        key=lambda path: (len(path.parts), str(path)),
    )

    if not matches:
        visible_txt = sorted(DATA_DIR.rglob("*.txt"))
        visible_text = "\n".join(f"  {path}" for path in visible_txt[:100])
        raise FileNotFoundError(
            f"Cannot find {filename} under {DATA_DIR}.\n"
            f"Visible TXT files (up to 100):\n{visible_text}"
        )

    if len(matches) > 1:
        print(f"[Path warning] Multiple matches for {filename}; using:")
        print("  ", matches[0])

    return matches[0]


def discover_data_paths():
    train_name = f"train_{RAW_SPECIES}_31.txt"
    valid_candidates = [
        f"valid_{RAW_SPECIES}_31.txt",
        f"val_{RAW_SPECIES}_31.txt",
        f"dataset_{RAW_SPECIES}_31.txt",
    ]
    test_name = f"test_{RAW_SPECIES}_31.txt"

    train_path = find_local_file(train_name)

    valid_path = None
    for candidate in valid_candidates:
        try:
            valid_path = find_local_file(candidate)
            break
        except FileNotFoundError:
            continue

    if valid_path is None:
        raise FileNotFoundError(
            "Validation file was not found under "
            f"{DATA_DIR}. Tried: {', '.join(valid_candidates)}"
        )

    # TEST is required because this version reports manuscript-compatible
    # reference metrics for every validation-selected checkpoint.
    test_path = find_local_file(test_name)

    paths = {
        "train": train_path,
        "val": valid_path,
        "test": test_path,
    }

    print("\nDiscovered local data paths:")
    for key, value in paths.items():
        print(f"  {key:<13s}: {value}")

    return paths


DATA_PATHS = discover_data_paths()


# ==========================================================
# 6) Amino-acid constants and BLOSUM62
# ==========================================================
AA20 = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]
AA21 = AA20 + ["X"]
AA_INDEX = {aa: index for index, aa in enumerate(AA21)}
AA20_SET = set(AA20)

KYTE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

FLEX = {
    "A": 0.357, "R": 0.529, "N": 0.463, "D": 0.511,
    "C": 0.346, "Q": 0.493, "E": 0.497, "G": 0.544,
    "H": 0.323, "I": 0.462, "L": 0.365, "K": 0.466,
    "M": 0.295, "F": 0.314, "P": 0.509, "S": 0.507,
    "T": 0.444, "W": 0.305, "Y": 0.420, "V": 0.386,
}

AROMATIC = {"F", "W", "Y", "H"}
POSITIVE = {"K", "R", "H"}
NEGATIVE = {"D", "E"}
POLAR = {"S", "T", "N", "Q", "Y", "C", "H"}
CHARGED = POSITIVE.union(NEGATIVE)

BLOSUM62 = {
    "A": [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
    "R": [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
    "N": [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
    "D": [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
    "C": [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    "Q": [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
    "E": [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
    "G": [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
    "H": [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
    "I": [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
    "L": [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
    "K": [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
    "M": [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
    "F": [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
    "P": [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
    "S": [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
    "T": [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
    "W": [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
    "Y": [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
    "V": [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
}

CONT_IDX = [42, 48, 49]
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY\*X]{31}$", re.IGNORECASE)


# ==========================================================
# 7) Robust TXT reader
# ==========================================================
def read_txt_auto(path, split_name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line)

    if not rows:
        raise ValueError(f"Empty file: {path}")

    parts = [re.split(r"[,\t ]+", row) for row in rows]
    n_columns = max(len(items) for items in parts)
    parts = [items + [""] * (n_columns - len(items)) for items in parts]
    raw_df = pd.DataFrame(parts)

    sequence_column = None
    for column in range(n_columns):
        sample_values = raw_df[column].astype(str).head(30)
        valid_count = sum(
            1
            for value in sample_values
            if len(value.strip()) == SEQ_LEN and AA_RE.match(value.strip())
        )
        if valid_count >= min(5, len(sample_values)):
            sequence_column = column
            break

    if sequence_column is None:
        raise ValueError(f"Cannot infer sequence column in: {path}")

    valid_label_tokens = {
        "0", "1", "true", "false", "pos", "neg",
        "positive", "negative", "yes", "no",
    }

    label_column = None
    for column in range(n_columns - 1, -1, -1):
        if column == sequence_column:
            continue
        values = raw_df[column].astype(str).str.strip().str.lower()
        unique_values = set(values.unique())
        if unique_values and unique_values.issubset(valid_label_tokens):
            label_column = column
            break

    if label_column is None:
        raise ValueError(f"Cannot infer label column in: {path}")

    data = pd.DataFrame({
        "seq": raw_df[sequence_column].astype(str).str.strip().str.upper(),
        "label": raw_df[label_column].astype(str).str.strip().str.lower(),
    })

    label_mapping = {
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "pos": 1, "neg": 0,
        "positive": 1, "negative": 0,
        "yes": 1, "no": 0,
    }
    data["label"] = data["label"].map(label_mapping)

    invalid_label_count = int(data["label"].isna().sum())
    if invalid_label_count:
        raise ValueError(
            f"{path}: {invalid_label_count} rows contain unrecognized labels."
        )

    data["label"] = data["label"].astype(int)

    before = len(data)
    valid_mask = (
        data["seq"].map(len).eq(SEQ_LEN)
        & data["seq"].map(lambda value: AA_RE.match(value) is not None)
    )
    data = data[valid_mask].reset_index(drop=True)

    dropped = before - len(data)
    if dropped:
        print(f"[{split_name}] dropped {dropped} invalid sequence rows")

    if data.empty:
        raise ValueError(f"No valid sequences remained in {path}")

    class_counts = data["label"].value_counts().sort_index().to_dict()
    print(
        f"[OK] {split_name:<5s} | rows={len(data):6d} | "
        f"labels={class_counts} | {path}"
    )
    return data


# ==========================================================
# 8) Feature construction and caching
# ==========================================================
def one_hot21(aa):
    vector = np.zeros(21, dtype=np.float32)
    normalized_aa = aa if aa in AA20_SET else "X"
    vector[AA_INDEX[normalized_aa]] = 1.0
    return vector


def blosum20_vec(aa):
    normalized_aa = aa if aa in AA20_SET else "X"
    return np.asarray(
        BLOSUM62.get(normalized_aa, [0] * 20),
        dtype=np.float32,
    )


def physchem8_vec(aa):
    if aa not in AA20_SET:
        return np.zeros(8, dtype=np.float32)

    return np.asarray([
        KYTE[aa],
        1.0 if aa in POLAR else 0.0,
        1.0 if aa in CHARGED else 0.0,
        1.0 if aa in POSITIVE else 0.0,
        1.0 if aa in NEGATIVE else 0.0,
        1.0 if aa in AROMATIC else 0.0,
        VOLUME[aa],
        FLEX[aa],
    ], dtype=np.float32)


def make_features_for_seq(sequence):
    if len(sequence) != SEQ_LEN:
        raise ValueError(
            f"Sequence length must be {SEQ_LEN}, got {len(sequence)}"
        )

    residue_features = []
    for aa in sequence:
        residue_features.append(
            np.concatenate([
                one_hot21(aa),
                blosum20_vec(aa),
                physchem8_vec(aa),
            ])
        )

    features_49 = np.stack(residue_features, axis=0)
    center = SEQ_LEN // 2
    dnorm = (
        (np.arange(SEQ_LEN, dtype=np.float32) - center)
        / (SEQ_LEN / 2.0)
    ).reshape(-1, 1)

    # 21 one-hot + 1 Dnorm + 20 BLOSUM62 + 8 PCP = 50.
    features_50 = np.concatenate([
        features_49[:, :21],
        dnorm,
        features_49[:, 21:],
    ], axis=1)

    return features_50.astype(np.float32)


def dataframe_to_feature_arrays(dataframe, split_name):
    sequences = dataframe["seq"].astype(str).tolist()
    labels = dataframe["label"].astype(int).to_numpy(dtype=np.int64)

    features = np.empty(
        (len(sequences), SEQ_LEN, INPUT_DIM),
        dtype=np.float32,
    )

    for index, sequence in enumerate(sequences):
        features[index] = make_features_for_seq(sequence)
        if (index + 1) % 2000 == 0 or index + 1 == len(sequences):
            print(
                f"[{split_name}] feature encoding: "
                f"{index + 1}/{len(sequences)}"
            )

    return features, labels


def source_signature(paths):
    items = []
    for key in ["train", "val", "test"]:
        path = Path(paths[key])
        stat = path.stat()
        items.append({
            "key": key,
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    payload = json.dumps(items, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_or_build_feature_cache():
    cache_file = CACHE_DIR / f"{RAW_SPECIES}_train_val_test_features.npz"
    metadata_file = CACHE_DIR / f"{RAW_SPECIES}_cache_meta_with_test.json"
    signature = source_signature(DATA_PATHS)

    if cache_file.exists() and metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata.get("source_signature") == signature:
                print("\nLoading feature cache:", cache_file)
                cache = np.load(cache_file, allow_pickle=False)
                return (
                    cache["X_train"].astype(np.float32),
                    cache["y_train"].astype(np.int64),
                    cache["X_val"].astype(np.float32),
                    cache["y_val"].astype(np.int64),
                    cache["X_test"].astype(np.float32),
                    cache["y_test"].astype(np.int64),
                )
        except Exception as error:
            print("Warning: failed to load existing feature cache:", error)

    print("\nBuilding TRAIN/VALID/TEST feature cache from TXT files...")
    train_df = read_txt_auto(DATA_PATHS["train"], "train")
    val_df = read_txt_auto(DATA_PATHS["val"], "val")
    test_df = read_txt_auto(DATA_PATHS["test"], "test")

    X_train, y_train = dataframe_to_feature_arrays(train_df, "train")
    X_val, y_val = dataframe_to_feature_arrays(val_df, "val")
    X_test, y_test = dataframe_to_feature_arrays(test_df, "test")

    np.savez_compressed(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    metadata = {
        "display_species": DISPLAY_SPECIES,
        "raw_species": RAW_SPECIES,
        "source_signature": signature,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "created_at": str(datetime.now(timezone.utc)),
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Saved feature cache:", cache_file)
    return X_train, y_train, X_val, y_val, X_test, y_test


(
    X_TRAIN_RAW,
    Y_TRAIN,
    X_VAL_RAW,
    Y_VAL,
    X_TEST_RAW,
    Y_TEST,
) = load_or_build_feature_cache()


# ==========================================================
# 9) Train-only feature normalization
# ==========================================================
def compute_channel_stats(X_train_raw):
    mean = np.zeros(INPUT_DIM, dtype=np.float32)
    std = np.ones(INPUT_DIM, dtype=np.float32)

    continuous_values = X_train_raw[:, :, CONT_IDX].reshape(-1, len(CONT_IDX))
    continuous_mean = continuous_values.mean(axis=0)
    continuous_std = continuous_values.std(axis=0)
    continuous_std = np.maximum(continuous_std, 1e-6)

    mean[CONT_IDX] = continuous_mean.astype(np.float32)
    std[CONT_IDX] = continuous_std.astype(np.float32)
    return mean, std


def normalize_features(features, mean, std):
    normalized = features.copy().astype(np.float32)
    normalized[:, :, CONT_IDX] = (
        normalized[:, :, CONT_IDX] - mean[CONT_IDX]
    ) / (std[CONT_IDX] + 1e-6)
    return normalized


CHANNEL_MEAN, CHANNEL_STD = compute_channel_stats(X_TRAIN_RAW)
X_TRAIN = normalize_features(X_TRAIN_RAW, CHANNEL_MEAN, CHANNEL_STD)
X_VAL = normalize_features(X_VAL_RAW, CHANNEL_MEAN, CHANNEL_STD)
X_TEST = normalize_features(X_TEST_RAW, CHANNEL_MEAN, CHANNEL_STD)

np.savez(
    RESULTS_DIR / "train_normalization_stats.npz",
    mean=CHANNEL_MEAN,
    std=CHANNEL_STD,
)

print("\nPrepared arrays:")
print("  X_train:", X_TRAIN.shape, "| y_train:", Y_TRAIN.shape)
print("  X_val  :", X_VAL.shape, "| y_val  :", Y_VAL.shape)
print("  X_test :", X_TEST.shape, "| y_test :", Y_TEST.shape)


# ==========================================================
# 10) Dataset and deterministic DataLoaders
# ==========================================================
class ArrayDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


TRAIN_DATASET = ArrayDataset(X_TRAIN, Y_TRAIN)
VAL_DATASET = ArrayDataset(X_VAL, Y_VAL)
TEST_DATASET = ArrayDataset(X_TEST, Y_TEST)


def seed_worker(worker_id):
    worker_seed = TRAIN_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_train_loader(batch_size, epoch):
    generator = torch.Generator()
    generator.manual_seed(TRAIN_SEED * 100000 + int(epoch))

    return DataLoader(
        TRAIN_DATASET,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )


def build_eval_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )


def build_val_loader(batch_size):
    return build_eval_loader(VAL_DATASET, batch_size)


def build_test_loader(batch_size):
    return build_eval_loader(TEST_DATASET, batch_size)


# ==========================================================
# 11) Full CenRes-Ace model
# ==========================================================
class SE1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, mid, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Spatial-attention kernel must be odd.")
        self.conv = nn.Conv1d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        average_map = torch.mean(x, dim=1, keepdim=True)
        maximum_map = torch.max(x, dim=1, keepdim=True)[0]
        weights = torch.sigmoid(
            self.conv(torch.cat([average_map, maximum_map], dim=1))
        )
        return x * weights


class SpatialDropout1D(nn.Dropout2d):
    def forward(self, x):
        # (B, C, L) -> (B, C, L, 1) -> spatial channel dropout.
        return super().forward(x.unsqueeze(-1)).squeeze(-1)


class MultiScaleResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        se_reduction,
        spatial_kernel,
    ):
        super().__init__()

        branch_channels = out_channels // 3
        last_branch_channels = out_channels - 2 * branch_channels

        self.branches = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
            ),
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                dilation=1,
            ),
            nn.Conv1d(
                in_channels,
                last_branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
        ])

        self.group_norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        self.channel_attention = SE1D(
            out_channels,
            reduction=se_reduction,
        )
        self.spatial_attention = SpatialAttention1D(
            kernel_size=spatial_kernel,
        )
        self.projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dropout = SpatialDropout1D(float(dropout))

    def forward(self, x):
        branch_outputs = [
            self.activation(branch(x))
            for branch in self.branches
        ]
        y = torch.cat(branch_outputs, dim=1)
        y = self.activation(
            self.group_norm(y) + self.projection(x)
        )
        y = self.channel_attention(y)
        y = self.spatial_attention(y)
        y = self.dropout(y)
        return y


class CenResAceModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.center_window = int(hparams["center_window"])
        self.stem = nn.Conv1d(INPUT_DIM, 64, kernel_size=1)
        self.stem_dropout = SpatialDropout1D(
            float(hparams["stem_dropout"])
        )

        self.block1 = MultiScaleResBlock(
            64,
            128,
            dropout=hparams["block_dropout"],
            se_reduction=FIXED_MODEL_CONFIG["se_reduction"],
            spatial_kernel=FIXED_MODEL_CONFIG["spatial_kernel"],
        )
        self.block2 = MultiScaleResBlock(
            128,
            192,
            dropout=hparams["block_dropout"],
            se_reduction=FIXED_MODEL_CONFIG["se_reduction"],
            spatial_kernel=FIXED_MODEL_CONFIG["spatial_kernel"],
        )
        self.block3 = MultiScaleResBlock(
            192,
            256,
            dropout=hparams["block_dropout"],
            se_reduction=FIXED_MODEL_CONFIG["se_reduction"],
            spatial_kernel=FIXED_MODEL_CONFIG["spatial_kernel"],
        )

        self.pre_rnn_norm = nn.LayerNorm(256)
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=int(hparams["rnn_hidden"]),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        feature_dim = 2 * int(hparams["rnn_hidden"])
        self.head = nn.Sequential(
            nn.Linear(feature_dim, int(hparams["mlp_hidden"])),
            nn.SiLU(),
            nn.Dropout(float(hparams["mlp_dropout"])),
            nn.Linear(int(hparams["mlp_hidden"]), 1),
        )

    def forward(self, x):
        # x: (B, L, C)
        x = x.transpose(1, 2)
        x = F.silu(self.stem(x))
        x = self.stem_dropout(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.transpose(1, 2)
        x = self.pre_rnn_norm(x)
        x, _ = self.rnn(x)

        _, sequence_length, _ = x.shape
        center = sequence_length // 2
        window = self.center_window

        positions = torch.arange(
            center - window,
            center + window + 1,
            device=x.device,
        ).clamp(0, sequence_length - 1)

        neighborhood = x[:, positions, :]

        distances = torch.arange(
            -window,
            window + 1,
            device=x.device,
            dtype=x.dtype,
        )
        gaussian_weights = torch.exp(
            -0.5 * (distances / 2.0) ** 2
        )
        gaussian_weights = (
            gaussian_weights / gaussian_weights.sum()
        ).view(1, -1, 1)

        pooled = (neighborhood * gaussian_weights).sum(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits


def count_trainable_parameters(model):
    return int(sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ))


# ==========================================================
# 12) AMP, loss, EMA and training utilities
# ==========================================================
def create_grad_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
        except TypeError:
            return torch.amp.GradScaler(enabled=AMP_ENABLED)
    return torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)


def autocast_context():
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=AMP_ENABLED)
        except TypeError:
            return torch.amp.autocast(enabled=AMP_ENABLED)
    return torch.cuda.amp.autocast(enabled=AMP_ENABLED)


def build_pos_weight(labels):
    positive = max(1, int(np.sum(labels == 1)))
    negative = max(1, int(np.sum(labels == 0)))
    ratio = negative / positive
    max_weight = FIXED_MODEL_CONFIG["max_pos_weight"]
    if max_weight is not None:
        ratio = min(float(max_weight), float(ratio))
    return float(ratio)


def weighted_bce_with_label_smoothing(
    logits,
    targets,
    pos_weight,
    smoothing,
):
    targets = targets.float()
    smoothing = float(smoothing)
    if smoothing > 0:
        targets = targets * (1.0 - smoothing) + 0.5 * smoothing

    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
    )


@torch.no_grad()
def update_ema_model(ema_model, online_model, decay):
    online_parameters = dict(online_model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        source_parameter = online_parameters[name]
        ema_parameter.mul_(decay).add_(
            source_parameter,
            alpha=1.0 - decay,
        )

    online_buffers = dict(online_model.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        source_buffer = online_buffers[name]
        ema_buffer.copy_(source_buffer)


def train_one_epoch(
    model,
    ema_model,
    loader,
    optimizer,
    scaler,
    pos_weight,
    label_smoothing,
    ema_decay,
):
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            logits = model(features)
            loss = weighted_bce_with_label_smoothing(
                logits,
                labels,
                pos_weight=pos_weight,
                smoothing=label_smoothing,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        update_ema_model(
            ema_model,
            model,
            decay=float(ema_decay),
        )

        total_loss += float(loss.item()) * labels.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def collect_scores(model, loader):
    model.eval()
    all_labels = []
    all_scores = []

    for features, labels in loader:
        features = features.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = model(features)
            probabilities = torch.sigmoid(logits)

        all_labels.append(labels.numpy().reshape(-1))
        all_scores.append(
            probabilities.detach().cpu().numpy().reshape(-1)
        )

    return (
        np.concatenate(all_labels).astype(np.int64),
        np.concatenate(all_scores).astype(np.float64),
    )


# ==========================================================
# 12) Metrics and the two threshold conventions
# ==========================================================
def safe_auc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def safe_auprc(y_true, scores):
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def expected_calibration_error(y_true, scores, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    if len(y_true) == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for index in range(int(n_bins)):
        left, right = bin_edges[index], bin_edges[index + 1]
        if index == int(n_bins) - 1:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(scores[mask]))
        observed = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(observed - confidence)
    return float(ece)


def threshold_at_specificity_original_interp(y_true, scores, target_sp=0.90):
    """Exact manuscript fixed-specificity convention: interpolate threshold on the same split."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, _, thresholds = roc_curve(y_true, scores)
    specificity = 1.0 - fpr

    order = np.argsort(specificity)
    specificity = specificity[order]
    thresholds = thresholds[order]

    insertion = np.searchsorted(specificity, target_sp)
    if insertion <= 0:
        threshold = float(thresholds[0])
    elif insertion >= len(specificity):
        threshold = float(thresholds[-1])
    else:
        sp0, sp1 = specificity[insertion - 1], specificity[insertion]
        th0, th1 = thresholds[insertion - 1], thresholds[insertion]
        if sp1 == sp0:
            threshold = float((th0 + th1) / 2.0)
        else:
            weight = (target_sp - sp0) / (sp1 - sp0)
            threshold = float(th0 + weight * (th1 - th0))

    if np.isinf(threshold):
        threshold = float(np.max(scores) + 1e-6)
    return threshold


def find_threshold_at_specificity(y_true, scores, target_sp=0.90):
    """Validation convention: maximize Sn among discrete points with Sp>=target."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    specificity = 1.0 - fpr

    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    specificity = specificity[finite]
    tpr = tpr[finite]

    eligible = np.where(specificity >= target_sp)[0]
    if eligible.size:
        best_local = eligible[np.argmax(tpr[eligible])]
        return float(thresholds[best_local])

    negative_scores = scores[y_true == 0]
    if negative_scores.size:
        return float(np.quantile(negative_scores, target_sp))
    return 0.5


def compute_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    try:
        mcc = float(matthews_corrcoef(y_true, predictions))
    except Exception:
        mcc = float("nan")

    try:
        brier = float(brier_score_loss(y_true, scores))
    except Exception:
        brier = float("nan")

    sn = float(recall_score(y_true, predictions, zero_division=0))
    return {
        "SN": sn,
        "REC": sn,
        "SP": float(tn / max(1, tn + fp)),
        "ACC": float(accuracy_score(y_true, predictions)),
        "PRE": float(precision_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "MCC": mcc,
        "AUC": safe_auc(y_true, scores),
        "AUPRC": safe_auprc(y_true, scores),
        "PRC": safe_auprc(y_true, scores),
        "Brier": brier,
        "ECE": expected_calibration_error(y_true, scores, ECE_N_BINS),
        "Threshold": float(threshold),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# ==========================================================
# 14) Search-plan generation and stable trial naming
# ==========================================================
def jsonable_hparams(hparams):
    return {
        key: (
            float(value)
            if isinstance(value, np.floating)
            else int(value)
            if isinstance(value, np.integer)
            else value
        )
        for key, value in hparams.items()
    }


def hparam_signature(hparams):
    payload = json.dumps(
        jsonable_hparams(hparams),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def make_trial_name(trial_index, hparams):
    return (
        f"trial_{trial_index:04d}_"
        f"b{hparams['batch_size']}_"
        f"lr{hparams['lr']:.0e}_"
        f"wd{hparams['weight_decay']:.0e}_"
        f"rh{hparams['rnn_hidden']}_"
        f"cw{hparams['center_window']}_"
        f"sd{hparams['stem_dropout']:.2f}_"
        f"bd{hparams['block_dropout']:.2f}_"
        f"mh{hparams['mlp_hidden']}_"
        f"md{hparams['mlp_dropout']:.2f}_"
        f"ls{hparams['label_smoothing']:.2f}_"
        f"ema{hparams['ema_decay']:.3f}_"
        f"p{hparams['early_stop_patience']}_"
        f"{hparam_signature(hparams)}"
    ).replace("+", "")


def total_cartesian_size(space):
    size = 1
    for values in space.values():
        size *= len(values)
    return int(size)


def sample_unique_random_combinations(space, n_trials, seed):
    maximum = total_cartesian_size(space)
    requested = min(int(n_trials), maximum)
    rng = random.Random(seed)
    combinations = []
    seen = set()

    while len(combinations) < requested:
        candidate = {
            key: rng.choice(list(values))
            for key, values in space.items()
        }
        signature = json.dumps(
            jsonable_hparams(candidate),
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen:
            continue
        seen.add(signature)
        combinations.append(candidate)

    return combinations


def generate_or_load_search_plan():
    plan_path = RESULTS_DIR / "search_plan.json"

    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        print("\nLoaded frozen search plan:", plan_path)
        return plan["trials"]

    sampled = sample_unique_random_combinations(
        HPARAM_SPACE,
        MAX_TRIALS,
        SEARCH_SEED,
    )

    trials = []
    seen_signatures = set()

    if INCLUDE_CURRENT_BASELINE:
        baseline = jsonable_hparams(CURRENT_BASELINE)
        signature = hparam_signature(baseline)
        seen_signatures.add(signature)
        trials.append({
            "trial": 1,
            "source": "current_baseline",
            "hparams": baseline,
        })

    for candidate in sampled:
        candidate = jsonable_hparams(candidate)
        signature = hparam_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        trials.append({
            "trial": len(trials) + 1,
            "source": "random_search",
            "hparams": candidate,
        })
        if len(trials) >= MAX_TRIALS:
            break

    plan = {
        "display_species": DISPLAY_SPECIES,
        "raw_species": RAW_SPECIES,
        "search_seed": SEARCH_SEED,
        "train_seed": TRAIN_SEED,
        "max_trials": MAX_TRIALS,
        "cartesian_space_size": total_cartesian_size(HPARAM_SPACE),
        "fixed_model_config": FIXED_MODEL_CONFIG,
        "hparam_space": HPARAM_SPACE,
        "trials": trials,
        "created_at": str(datetime.now(timezone.utc)),
    }

    plan_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSaved frozen search plan:", plan_path)
    return trials


SEARCH_PLAN = generate_or_load_search_plan()


# ==========================================================
# 15) Checkpoint and serialization helpers
# ==========================================================
def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as error:
        print("Warning: failed to restore RNG state:", error)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_json(data, path):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def save_last_checkpoint(
    path,
    epoch,
    model,
    ema_model,
    optimizer,
    scheduler,
    scaler,
    best_auc,
    best_auprc,
    best_epoch,
    bad_epochs,
    history,
    hparams,
):
    torch.save({
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_auc": float(best_auc),
        "best_auprc": float(best_auprc),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "history": history,
        "hparams": hparams,
        "rng_state": capture_rng_state(),
    }, path)


# ==========================================================
# 16) Plot and table helpers
# ==========================================================
def plot_training_history(history, out_path, title):
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    auc_values = [row["val_AUC"] for row in history]
    auprc_values = [row["val_AUPRC"] for row in history]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(epochs, auc_values, marker="o", markersize=2.5, label="Validation AUROC")
    ax.plot(epochs, auprc_values, marker="s", markersize=2.5, label="Validation AUPRC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_score_curves(y_true, scores, trial_dir, prefix, title_prefix):
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr,
        "Threshold": roc_thresholds,
    }).to_csv(trial_dir / f"{prefix}_roc_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(
        fpr,
        tpr,
        linewidth=1.8,
        label=f"AUROC = {safe_auc(y_true, scores):.4f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title_prefix} ROC")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(trial_dir / f"{prefix}_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    threshold_column = np.concatenate([pr_thresholds, [np.nan]])
    pd.DataFrame({
        "Recall": recall,
        "Precision": precision,
        "Threshold": threshold_column,
    }).to_csv(trial_dir / f"{prefix}_pr_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(
        recall,
        precision,
        linewidth=1.8,
        label=f"AUPRC = {safe_auprc(y_true, scores):.4f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title_prefix} PR curve")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(trial_dir / f"{prefix}_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_validation_curves(y_true, scores, trial_dir):
    save_score_curves(
        y_true,
        scores,
        trial_dir,
        prefix="val",
        title_prefix=f"{DISPLAY_SPECIES} validation",
    )


def save_test_curves(y_true, scores, trial_dir):
    save_score_curves(
        y_true,
        scores,
        trial_dir,
        prefix="test",
        title_prefix=f"{DISPLAY_SPECIES} test",
    )


def save_rows_csv(rows, path):
    if not rows:
        return
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(path, index=False)


def valid_success_rows(rows):
    return [
        row
        for row in rows
        if row.get("status") == "ok"
        and np.isfinite(float(row.get("best_val_AUC", np.nan)))
    ]


def rebuild_summary_tables():
    rows = []

    for summary_path in sorted(TRIALS_DIR.glob("*/trial_summary.json")):
        try:
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except Exception as error:
            print("Warning: failed to read", summary_path, error)

    for failed_path in sorted(TRIALS_DIR.glob("*/failed_trial.json")):
        try:
            rows.append(json.loads(failed_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    if not rows:
        return []

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(RESULTS_DIR / "summary_all_trials.csv", index=False)
    dataframe.to_csv(RESULTS_DIR / "summary_all_trials_so_far.csv", index=False)

    try:
        dataframe.to_excel(RESULTS_DIR / "summary_all_trials.xlsx", index=False)
    except Exception as error:
        print("Warning: Excel export failed:", error)

    ok_dataframe = dataframe[dataframe["status"] == "ok"].copy()
    if ok_dataframe.empty:
        return rows

    numeric_columns = [
        "best_val_AUC", "best_val_AUPRC", "val_SN", "val_SP",
        "val_ACC", "val_PRE", "val_F1", "val_MCC",
        "test_SN", "test_SP", "test_ACC", "test_PRE", "test_F1",
        "test_MCC", "test_AUC", "test_AUPRC", "test_PRC",
        "test_valthr_SN", "test_valthr_SP", "test_valthr_ACC",
        "test_valthr_PRE", "test_valthr_F1", "test_valthr_MCC",
    ]
    for column in numeric_columns:
        if column in ok_dataframe.columns:
            ok_dataframe[column] = pd.to_numeric(ok_dataframe[column], errors="coerce")

    sort_specs = {
        "top10_by_val_AUC.csv": [
            "best_val_AUC", "best_val_AUPRC", "val_F1", "val_MCC",
        ],
        "top10_by_val_AUPRC.csv": [
            "best_val_AUPRC", "best_val_AUC", "val_F1", "val_MCC",
        ],
        "top10_by_val_F1.csv": [
            "val_F1", "best_val_AUC", "best_val_AUPRC", "val_MCC",
        ],
        "top10_by_val_MCC.csv": [
            "val_MCC", "best_val_AUC", "best_val_AUPRC", "val_F1",
        ],
        # The following TEST rankings reproduce the manuscript-compatible exploratory views.
        "top10_by_test_F1.csv": ["test_F1", "test_ACC", "test_AUC"],
        "top10_by_test_AUC.csv": ["test_AUC", "test_F1", "test_ACC"],
        "top10_by_test_ACC.csv": ["test_ACC", "test_F1", "test_AUC"],
        "top10_by_test_MCC.csv": ["test_MCC", "test_F1", "test_AUC"],
        "top10_by_test_AUPRC.csv": ["test_AUPRC", "test_AUC", "test_F1"],
    }

    for filename, columns in sort_specs.items():
        existing_columns = [column for column in columns if column in ok_dataframe.columns]
        if not existing_columns:
            continue
        top = ok_dataframe.sort_values(
            existing_columns,
            ascending=[False] * len(existing_columns),
            na_position="last",
        ).head(10)
        top.to_csv(RESULTS_DIR / filename, index=False)

    # Formal overall checkpoint remains selected only by validation results.
    best_by_auc = ok_dataframe.sort_values(
        ["best_val_AUC", "best_val_AUPRC", "val_F1", "val_MCC"],
        ascending=[False, False, False, False],
        na_position="last",
    ).iloc[0]

    best_checkpoint_path = Path(str(best_by_auc["checkpoint_path"]))
    if best_checkpoint_path.exists():
        best_object = safe_torch_load(best_checkpoint_path, map_location="cpu")
        best_object["selected_by"] = (
            "highest_validation_AUROC_then_AUPRC_across_stage1_trials"
        )
        best_object["top1_summary"] = best_by_auc.to_dict()
        best_object["test_rankings_are_exploratory"] = True
        torch.save(best_object, RESULTS_DIR / "best_overall_by_val_AUC.pt")

    print("\nCurrent Top 10 by validation AUROC:")
    display_columns = [
        "trial", "source", "best_epoch", "best_val_AUC",
        "best_val_AUPRC", "val_F1", "val_MCC",
        "test_AUC", "test_AUPRC", "test_ACC", "test_F1", "test_MCC",
        "batch_size", "lr", "weight_decay", "rnn_hidden",
        "center_window", "block_dropout", "mlp_dropout",
        "label_smoothing", "ema_decay",
    ]
    display_columns = [c for c in display_columns if c in ok_dataframe.columns]
    print(
        ok_dataframe.sort_values(
            ["best_val_AUC", "best_val_AUPRC", "val_F1", "val_MCC"],
            ascending=[False, False, False, False],
        )[display_columns].head(10).to_string(index=False)
    )

    return rows


# ==========================================================
# 17) Evaluate/backfill one validation-selected checkpoint
# ==========================================================
def summary_has_test_metrics(summary):
    return bool(
        summary.get("test_used", False)
        and "test_AUC" in summary
        and "test_F1" in summary
    )


def evaluate_trial_checkpoint(
    trial_index,
    source,
    hparams,
    trial_name,
    trial_dir,
    best_model_path,
    summary_path,
    parameter_count=None,
    elapsed_seconds=None,
):
    best_object = safe_torch_load(best_model_path, map_location="cpu")
    best_model = CenResAceModel(hparams).to(DEVICE)
    best_model.load_state_dict(best_object["model_state"])

    if parameter_count is None:
        parameter_count = count_trainable_parameters(best_model)

    val_loader = build_val_loader(hparams["batch_size"])
    test_loader = build_test_loader(hparams["batch_size"])

    val_labels, val_scores = collect_scores(best_model, val_loader)
    val_threshold = find_threshold_at_specificity(
        val_labels,
        val_scores,
        TARGET_SPECIFICITY,
    )
    val_metrics = compute_metrics(val_labels, val_scores, val_threshold)

    test_labels, test_scores = collect_scores(best_model, test_loader)

    # Main test_* columns: exact manuscript-compatible same-TEST threshold.
    if OLD_TEST_THRESHOLD_MODE != "original_interp":
        raise ValueError(
            f"Unsupported OLD_TEST_THRESHOLD_MODE={OLD_TEST_THRESHOLD_MODE!r}"
        )
    test_threshold = threshold_at_specificity_original_interp(
        test_labels,
        test_scores,
        TARGET_SPECIFICITY,
    )
    test_metrics = compute_metrics(test_labels, test_scores, test_threshold)

    # Additional fairer convention: apply the validation-derived threshold.
    test_valthr_metrics = compute_metrics(
        test_labels,
        test_scores,
        val_threshold,
    )

    pd.DataFrame({
        "y_true": val_labels,
        "y_score": val_scores,
        "y_pred": (val_scores >= val_threshold).astype(int),
        "threshold": float(val_threshold),
    }).to_csv(trial_dir / "val_predictions.csv", index=False)

    pd.DataFrame({
        "y_true": test_labels,
        "y_score": test_scores,
        "y_pred_old_test_sp_interp": (
            test_scores >= test_threshold
        ).astype(int),
        "y_pred_validation_threshold": (
            test_scores >= val_threshold
        ).astype(int),
        "old_test_sp_interp_threshold": float(test_threshold),
        "validation_threshold": float(val_threshold),
    }).to_csv(trial_dir / "test_predictions.csv", index=False)

    save_validation_curves(val_labels, val_scores, trial_dir)
    save_test_curves(test_labels, test_scores, trial_dir)

    history_path = trial_dir / "history.csv"
    if history_path.exists():
        try:
            history = pd.read_csv(history_path).to_dict(orient="records")
            plot_training_history(
                history,
                trial_dir / "validation_history.png",
                title=f"{DISPLAY_SPECIES} trial {trial_index:04d}",
            )
        except Exception as error:
            print("Warning: history plot could not be rebuilt:", error)

    existing_summary = {}
    if summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            existing_summary = {}

    if elapsed_seconds is None:
        elapsed_seconds = float(existing_summary.get("elapsed_seconds", 0.0))

    summary = {
        "status": "ok",
        "error": "",
        "Species": DISPLAY_SPECIES,
        "raw_species": RAW_SPECIES,
        "trial": int(trial_index),
        "source": source,
        "trial_name": trial_name,
        "best_epoch": int(best_object["best_epoch"]),
        "best_val_AUC": float(val_metrics["AUC"]),
        "best_val_AUPRC": float(val_metrics["AUPRC"]),
        "val_SN": float(val_metrics["SN"]),
        "val_SP": float(val_metrics["SP"]),
        "val_ACC": float(val_metrics["ACC"]),
        "val_PRE": float(val_metrics["PRE"]),
        "val_F1": float(val_metrics["F1"]),
        "val_MCC": float(val_metrics["MCC"]),
        "val_AUC": float(val_metrics["AUC"]),
        "val_AUPRC": float(val_metrics["AUPRC"]),
        "val_PRC": float(val_metrics["PRC"]),
        "val_Brier": float(val_metrics["Brier"]),
        "val_ECE": float(val_metrics["ECE"]),
        "val_Threshold@Sp0.90": float(val_metrics["Threshold"]),
        "val_TP": int(val_metrics["TP"]),
        "val_TN": int(val_metrics["TN"]),
        "val_FP": int(val_metrics["FP"]),
        "val_FN": int(val_metrics["FN"]),

        # Manuscript-compatible TEST metrics: threshold derived on TEST itself.
        "test_SN": float(test_metrics["SN"]),
        "test_REC": float(test_metrics["REC"]),
        "test_SP": float(test_metrics["SP"]),
        "test_ACC": float(test_metrics["ACC"]),
        "test_PRE": float(test_metrics["PRE"]),
        "test_F1": float(test_metrics["F1"]),
        "test_MCC": float(test_metrics["MCC"]),
        "test_AUC": float(test_metrics["AUC"]),
        "test_AUPRC": float(test_metrics["AUPRC"]),
        "test_PRC": float(test_metrics["PRC"]),
        "test_Brier": float(test_metrics["Brier"]),
        "test_ECE": float(test_metrics["ECE"]),
        "test_Threshold@Sp": float(test_metrics["Threshold"]),
        "test_TP": int(test_metrics["TP"]),
        "test_TN": int(test_metrics["TN"]),
        "test_FP": int(test_metrics["FP"]),
        "test_FN": int(test_metrics["FN"]),

        # Same model/scores, but validation-derived threshold applied to TEST.
        "test_valthr_SN": float(test_valthr_metrics["SN"]),
        "test_valthr_SP": float(test_valthr_metrics["SP"]),
        "test_valthr_ACC": float(test_valthr_metrics["ACC"]),
        "test_valthr_PRE": float(test_valthr_metrics["PRE"]),
        "test_valthr_F1": float(test_valthr_metrics["F1"]),
        "test_valthr_MCC": float(test_valthr_metrics["MCC"]),
        "test_valthr_Threshold": float(val_threshold),
        "test_valthr_TP": int(test_valthr_metrics["TP"]),
        "test_valthr_TN": int(test_valthr_metrics["TN"]),
        "test_valthr_FP": int(test_valthr_metrics["FP"]),
        "test_valthr_FN": int(test_valthr_metrics["FN"]),

        "parameter_count": int(parameter_count),
        "train_rows": int(len(Y_TRAIN)),
        "val_rows": int(len(Y_VAL)),
        "test_rows": int(len(Y_TEST)),
        "elapsed_seconds": float(elapsed_seconds),
        "checkpoint_path": str(best_model_path),
        "trial_dir": str(trial_dir),
        "test_used": True,
        "test_used_for_hyperparameter_selection": False,
        "test_threshold_mode": OLD_TEST_THRESHOLD_MODE,
        "test_metrics_reference_only": True,
        "completed_at": str(datetime.now(timezone.utc)),
    }
    summary.update(hparams)
    save_json(summary, summary_path)

    # Add the evaluation records to the checkpoint without changing its model.
    best_object["val_metrics_recomputed"] = val_metrics
    best_object["test_metrics_old_test_sp_interp"] = test_metrics
    best_object["test_metrics_validation_threshold"] = test_valthr_metrics
    best_object["test_used_for_selection"] = False
    best_object["test_threshold_mode"] = OLD_TEST_THRESHOLD_MODE
    torch.save(best_object, best_model_path)

    print(
        f"[Trial {trial_index:04d}] EVALUATED | "
        f"VAL AUC={val_metrics['AUC']:.4f} | "
        f"TEST AUC={test_metrics['AUC']:.4f} | "
        f"TEST AUPRC={test_metrics['AUPRC']:.4f} | "
        f"TEST ACC={test_metrics['ACC']:.4f} | "
        f"TEST F1={test_metrics['F1']:.4f} | "
        f"TEST MCC={test_metrics['MCC']:.4f}"
    )

    del best_model, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ==========================================================
# 18) Train or resume one Stage-1 trial
# ==========================================================
def train_one_trial(trial_record):
    trial_index = int(trial_record["trial"])
    source = str(trial_record.get("source", "random_search"))
    hparams = jsonable_hparams(trial_record["hparams"])

    trial_name = make_trial_name(trial_index, hparams)
    trial_dir = TRIALS_DIR / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    summary_path = trial_dir / "trial_summary.json"
    failed_path = trial_dir / "failed_trial.json"
    best_model_path = trial_dir / "best_model_by_val_auc.pt"
    last_checkpoint_path = trial_dir / "last_checkpoint.pt"
    history_path = trial_dir / "history.csv"

    if summary_path.exists() and best_model_path.exists() and not FORCE_RERUN_COMPLETED:
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_has_test_metrics(existing_summary):
            print(
                f"[Trial {trial_index:04d}] already complete with TEST metrics "
                f"-> skipped: {trial_name}"
            )
            return existing_summary

        # Important: old completed trials are backfilled on TEST without retraining.
        print(
            f"[Trial {trial_index:04d}] training already complete but TEST "
            "columns are missing -> evaluating saved checkpoint only"
        )
        return evaluate_trial_checkpoint(
            trial_index=trial_index,
            source=source,
            hparams=hparams,
            trial_name=trial_name,
            trial_dir=trial_dir,
            best_model_path=best_model_path,
            summary_path=summary_path,
            elapsed_seconds=float(existing_summary.get("elapsed_seconds", 0.0)),
        )

    if failed_path.exists():
        failed_path.unlink()

    set_global_seed(TRAIN_SEED)
    val_loader = build_val_loader(hparams["batch_size"])

    model = CenResAceModel(hparams).to(DEVICE)
    ema_model = CenResAceModel(hparams).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=float(FIXED_MODEL_CONFIG["eta_min"]),
    )

    scaler = create_grad_scaler()
    positive_weight_value = build_pos_weight(Y_TRAIN)
    positive_weight = torch.tensor(
        [positive_weight_value],
        dtype=torch.float32,
        device=DEVICE,
    )

    start_epoch = 1
    best_auc = -float("inf")
    best_auprc = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    if last_checkpoint_path.exists():
        print(f"[Trial {trial_index:04d}] resuming from: {last_checkpoint_path}")
        checkpoint = safe_torch_load(last_checkpoint_path, map_location=DEVICE)
        checkpoint_hparams = jsonable_hparams(checkpoint.get("hparams", {}))
        if checkpoint_hparams != hparams:
            raise RuntimeError(
                "Existing last_checkpoint.pt belongs to different hparams. "
                f"Expected {hparams}, found {checkpoint_hparams}."
            )

        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        try:
            scaler.load_state_dict(checkpoint["scaler_state"])
        except Exception as error:
            print("Warning: GradScaler state was not restored:", error)

        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint["best_auc"])
        best_auprc = float(checkpoint["best_auprc"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = list(checkpoint.get("history", []))
        restore_rng_state(checkpoint.get("rng_state"))

    print("\n" + "#" * 110)
    print(f"[TRIAL {trial_index:04d}/{len(SEARCH_PLAN):04d}] {trial_name}")
    print("Source:", source)
    print("Hparams:")
    print(json.dumps(hparams, indent=2, ensure_ascii=False))
    print("Trainable parameters:", parameter_count)
    print("Positive weight:", positive_weight_value)
    print("Start epoch:", start_epoch)
    print("#" * 110)

    started_at = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        train_loader = build_train_loader(hparams["batch_size"], epoch)
        train_loss = train_one_epoch(
            model=model,
            ema_model=ema_model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            pos_weight=positive_weight,
            label_smoothing=hparams["label_smoothing"],
            ema_decay=hparams["ema_decay"],
        )

        val_labels, val_scores = collect_scores(ema_model, val_loader)
        val_auc = safe_auc(val_labels, val_scores)
        val_auprc = safe_auprc(val_labels, val_scores)
        scheduler.step()

        improved = False
        if np.isfinite(val_auc):
            if val_auc > best_auc + 1e-12:
                improved = True
            elif abs(val_auc - best_auc) <= 1e-12 and val_auprc > best_auprc + 1e-12:
                improved = True

        if improved:
            best_auc = float(val_auc)
            best_auprc = float(val_auprc)
            best_epoch = int(epoch)
            bad_epochs = 0

            threshold = find_threshold_at_specificity(
                val_labels,
                val_scores,
                TARGET_SPECIFICITY,
            )
            metrics_at_best = compute_metrics(val_labels, val_scores, threshold)

            torch.save({
                "display_species": DISPLAY_SPECIES,
                "raw_species": RAW_SPECIES,
                "trial": trial_index,
                "source": source,
                "hparams": hparams,
                "fixed_model_config": FIXED_MODEL_CONFIG,
                "best_epoch": best_epoch,
                "best_val_AUC": best_auc,
                "best_val_AUPRC": best_auprc,
                "val_metrics_at_best": metrics_at_best,
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in ema_model.state_dict().items()
                },
                "online_model_state": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "normalization_mean": CHANNEL_MEAN,
                "normalization_std": CHANNEL_STD,
                "parameter_count": parameter_count,
                "selected_by": "validation_AUROC_with_AUPRC_tie_break",
                "test_used": False,
                "test_used_for_hyperparameter_selection": False,
            }, best_model_path)
        else:
            bad_epochs += 1

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_AUC": float(val_auc),
            "val_AUPRC": float(val_auprc),
            "best_val_AUC": float(best_auc),
            "best_val_AUPRC": float(best_auprc),
            "best_epoch": int(best_epoch),
            "bad_epochs": int(bad_epochs),
            "lr": current_lr,
        })
        pd.DataFrame(history).to_csv(history_path, index=False)

        save_last_checkpoint(
            path=last_checkpoint_path,
            epoch=epoch,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_auc=best_auc,
            best_auprc=best_auprc,
            best_epoch=best_epoch,
            bad_epochs=bad_epochs,
            history=history,
            hparams=hparams,
        )

        if epoch == 1 or epoch % PRINT_EVERY == 0 or improved:
            print(
                f"[Trial {trial_index:04d}] epoch={epoch:03d} | "
                f"loss={train_loss:.5f} | "
                f"val AUC={val_auc:.4f} | val AUPRC={val_auprc:.4f} | "
                f"best={best_auc:.4f}/{best_auprc:.4f}@{best_epoch} | "
                f"bad={bad_epochs} | lr={current_lr:.2e}"
            )

        del train_loader
        gc.collect()

        if bad_epochs >= int(hparams["early_stop_patience"]):
            print(f"[Trial {trial_index:04d}] early stopping at epoch {epoch}.")
            break

    if not best_model_path.exists():
        raise RuntimeError(f"Trial {trial_index} did not produce a valid best checkpoint.")

    elapsed_seconds = float(time.time() - started_at)

    # Release training objects before final VALID/TEST inference.
    del model, ema_model, optimizer, scheduler, scaler, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = evaluate_trial_checkpoint(
        trial_index=trial_index,
        source=source,
        hparams=hparams,
        trial_name=trial_name,
        trial_dir=trial_dir,
        best_model_path=best_model_path,
        summary_path=summary_path,
        parameter_count=parameter_count,
        elapsed_seconds=elapsed_seconds,
    )

    if last_checkpoint_path.exists() and not KEEP_LAST_CHECKPOINT_AFTER_FINISH:
        last_checkpoint_path.unlink()

    return summary


# ==========================================================
# 18) Run metadata
# ==========================================================
def save_run_config():
    config = {
        "display_species": DISPLAY_SPECIES,
        "raw_species": RAW_SPECIES,
        "platform": sys.platform,
        "gpu_id": int(GPU_ID),
        "data_dir": str(DATA_DIR),
        "data_paths": {
            key: str(value) if value is not None else None
            for key, value in DATA_PATHS.items()
        },
        "work_root": str(WORK_ROOT),
        "search_mode": SEARCH_MODE,
        "max_trials": MAX_TRIALS,
        "include_current_baseline": INCLUDE_CURRENT_BASELINE,
        "search_seed": SEARCH_SEED,
        "train_seed": TRAIN_SEED,
        "max_epochs": MAX_EPOCHS,
        "target_specificity": TARGET_SPECIFICITY,
        "use_amp": AMP_ENABLED,
        "fixed_model_config": FIXED_MODEL_CONFIG,
        "current_baseline": CURRENT_BASELINE,
        "hparam_space": HPARAM_SPACE,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "test_used": True,
        "test_used_for_hyperparameter_selection": False,
        "test_threshold_mode": OLD_TEST_THRESHOLD_MODE,
        "report_validation_threshold_on_test": REPORT_VALIDATION_THRESHOLD_ON_TEST,
        "created_at": str(datetime.now(timezone.utc)),
    }
    save_json(config, RESULTS_DIR / "run_config.json")


save_run_config()


# ==========================================================
# 19) Main search loop
# ==========================================================
def main():
    print("\n" + "=" * 110)
    print("STAGE 1 RANDOM SEARCH START")
    print("=" * 110)
    print("Species:", DISPLAY_SPECIES)
    print("Trials in frozen plan:", len(SEARCH_PLAN))
    print("Cartesian-space size:", total_cartesian_size(HPARAM_SPACE))
    print("Primary ranking: validation AUROC -> AUPRC -> F1 -> MCC")
    print("TEST reference metrics are reported; formal ranking remains validation-based.")
    print("Main test_* threshold mode:", OLD_TEST_THRESHOLD_MODE)
    print("=" * 110)

    for position, trial_record in enumerate(SEARCH_PLAN, start=1):
        try:
            train_one_trial(trial_record)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Current epoch checkpoint has been saved.")
            rebuild_summary_tables()
            raise
        except Exception as error:
            trial_index = int(trial_record["trial"])
            hparams = jsonable_hparams(trial_record["hparams"])
            trial_name = make_trial_name(trial_index, hparams)
            trial_dir = TRIALS_DIR / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)

            error_text = repr(error)
            print(
                f"\n[ERROR] Trial {trial_index:04d} failed: {error_text}"
            )
            traceback.print_exc()

            failed_summary = {
                "status": "failed",
                "error": error_text,
                "Species": DISPLAY_SPECIES,
                "raw_species": RAW_SPECIES,
                "trial": trial_index,
                "source": trial_record.get("source", "random_search"),
                "trial_name": trial_name,
                "trial_dir": str(trial_dir),
                "test_used": False,
                "test_used_for_hyperparameter_selection": False,
                "failed_at": str(datetime.now(timezone.utc)),
            }
            failed_summary.update(hparams)
            save_json(
                failed_summary,
                trial_dir / "failed_trial.json",
            )

        finally:
            rebuild_summary_tables()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rebuild_summary_tables()

    print("\n" + "=" * 110)
    print("STAGE 1 RANDOM SEARCH FINISHED")
    print("=" * 110)
    print("All trials:", RESULTS_DIR / "summary_all_trials.csv")
    print("Top 10 by AUC:", RESULTS_DIR / "top10_by_val_AUC.csv")
    print("Top 10 by AUPRC:", RESULTS_DIR / "top10_by_val_AUPRC.csv")
    print("Top 10 by F1:", RESULTS_DIR / "top10_by_val_F1.csv")
    print("Top 10 by validation MCC:", RESULTS_DIR / "top10_by_val_MCC.csv")
    print("Exploratory Top 10 by TEST F1:", RESULTS_DIR / "top10_by_test_F1.csv")
    print("Exploratory Top 10 by TEST AUC:", RESULTS_DIR / "top10_by_test_AUC.csv")
    print("Exploratory Top 10 by TEST ACC:", RESULTS_DIR / "top10_by_test_ACC.csv")
    print("Best checkpoint selected by VALID AUC:", RESULTS_DIR / "best_overall_by_val_AUC.pt")
    print("Work root:", WORK_ROOT)
    print("=" * 110)


if __name__ == "__main__":
    main()