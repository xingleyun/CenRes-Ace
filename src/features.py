import numpy as np
from .config import SEQ_LEN

AA20 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
AA21 = AA20 + ['X']
AA_INDEX = {a:i for i,a in enumerate(AA21)}

KYTE = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,"H":-3.2,
        "I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
VOLUME = {"A":88.6,"R":173.4,"N":114.1,"D":111.1,"C":108.5,"Q":143.8,"E":138.4,"G":60.1,"H":153.2,
          "I":166.7,"L":166.7,"K":168.6,"M":162.9,"F":189.9,"P":112.7,"S":89.0,"T":116.1,"W":227.8,"Y":193.6,"V":140.0}
FLEX = {"A":0.357,"R":0.529,"N":0.463,"D":0.511,"C":0.346,"Q":0.493,"E":0.497,"G":0.544,"H":0.323,
        "I":0.462,"L":0.365,"K":0.466,"M":0.295,"F":0.314,"P":0.509,"S":0.507,"T":0.444,"W":0.305,"Y":0.420,"V":0.386}
AROMATIC = set(["F","W","Y","H"])
POSITIVE = set(["K","R","H"])
NEGATIVE = set(["D","E"])
POLAR    = set(["S","T","N","Q","Y","C","H"])
CHARGED  = POSITIVE.union(NEGATIVE)

BLOSUM62 = {
"A":[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
"R":[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
"N":[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
"D":[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
"C":[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
"Q":[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
"E":[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
"G":[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
"H":[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
"I":[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
"L":[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
"K":[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
"M":[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
"F":[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
"P":[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
"S":[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
"T":[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
"W":[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
"Y":[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
"V":[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
}

def one_hot21(aa: str):
    v = np.zeros(21, dtype=np.float32)
    v[AA_INDEX.get(aa, 20)] = 1.0
    return v

def blosum20_vec(aa: str):
    return np.array(BLOSUM62.get(aa, [0]*20), dtype=np.float32)

def physchem8_vec(aa: str):
    if aa not in AA20:
        return np.zeros(8, dtype=np.float32)
    hydro = KYTE[aa]
    is_polar = 1.0 if aa in POLAR else 0.0
    is_charged = 1.0 if aa in CHARGED else 0.0
    is_pos = 1.0 if aa in POSITIVE else 0.0
    is_neg = 1.0 if aa in NEGATIVE else 0.0
    is_arom = 1.0 if aa in AROMATIC else 0.0
    vol = VOLUME[aa]
    flex = FLEX[aa]
    return np.array([hydro, is_polar, is_charged, is_pos, is_neg, is_arom, vol, flex], dtype=np.float32)

def make_features_for_seq(seq: str):
    assert len(seq) == SEQ_LEN, f"Sequence length must be {SEQ_LEN}, got {len(seq)}"
    L = len(seq)
    rows = [np.concatenate([one_hot21(aa), blosum20_vec(aa), physchem8_vec(aa)], 0) for aa in seq]
    X = np.stack(rows, 0)
    center = L // 2
    dnorm = ((np.arange(L, dtype=np.float32) - center) / (L / 2.0)).reshape(-1, 1)
    X = np.concatenate([X[:, :21], dnorm, X[:, 21:]], 1).astype(np.float32)
    return X

CONT_IDX = [42, 48, 49]

def compute_channel_stats_for_train(df_train):
    C = 50
    s = np.zeros(C, np.float64)
    s2 = np.zeros(C, np.float64)
    n = 0
    for seq in df_train["seq"].values:
        X = make_features_for_seq(seq)
        vals = X[:, CONT_IDX]
        s[CONT_IDX] += vals.sum(0)
        s2[CONT_IDX] += (vals**2).sum(0)
        n += X.shape[0]

    mean = np.zeros(C, np.float32)
    std = np.ones(C, np.float32)
    if n > 0:
        m = s / n
        v = np.maximum(s2 / n - m * m, 1e-8)
        mean[CONT_IDX] = m[CONT_IDX].astype(np.float32)
        std[CONT_IDX] = np.sqrt(v[CONT_IDX]).astype(np.float32)
    return mean, std
