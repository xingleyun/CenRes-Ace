import os
import random
import numpy as np
import torch

# -------------------------
# 1) High-level config
# -------------------------
DATA_PATHS = {
    "train": "data/train_Rattus_norvegicus_31.txt",
    "val":   "data/valid_Rattus_norvegicus_31.txt",
    "test":  "data/test_Rattus_norvegicus_31.txt",
}
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# 2) Reproducibility / Determinism
# -------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print("Warning: deterministic_algorithms not fully available:", e)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_CUDA = torch.cuda.is_available()

# -------------------------
# 3) Training config
# -------------------------
SEQ_LEN     = 31
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 3e-4
WD          = 5e-4
NUM_WORKERS = 2
TARGET_SPECIFICITY = 0.90
USE_GRU     = True
CENTER_WINDOW = 5
LOSS_TYPE   = "bce_pos_weight"   # {"bce_pos_weight", "focal", "label_smooth"}
USE_SWA     = True
USE_EMA     = False
USE_LAYER_NORM_PRE_RNN = True
N_FOLDS     = 5
