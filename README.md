# CenRes-Ace

Official code repository for **CenRes-Ace: A Center-Aware Multi-Scale Residual CNN-BiGRU Framework for Species-Specific Lysine Acetylation-Site Prediction**.

CenRes-Ace combines residue-level sequence descriptors, multi-scale residual convolution, channel and spatial attention, BiGRU contextual encoding, and Gaussian center-aware aggregation for species-specific lysine acetylation-site prediction.

## Repository contents

```text
CenRes-Ace/
├── cenres_ace_hyperparameter_search.py
├── configs/
│   └── final_species_configs.json
├── docs/
│   └── DATA_LAYOUT.md
├── requirements.txt
├── .gitignore
└── README.md
```

The same Python script supports all nine species; separate copies of the model code are not required. The final species-specific configurations reported in the manuscript are provided in `configs/final_species_configs.json`.

## Model architecture

The full model uses:

- one-hot identity encoding, normalized relative position, BLOSUM62, and physicochemical descriptors;
- a 1x1 convolutional stem;
- three multi-scale residual CNN blocks;
- channel attention and spatial attention;
- a bidirectional GRU;
- Gaussian center-aware aggregation;
- an MLP classifier.

The sequence window length is 31 residues and the input dimension is 50.

## Installation

```bash
git clone https://github.com/xingleyun/CenRes-Ace.git
cd CenRes-Ace
pip install -r requirements.txt
```

## Data preparation

The benchmark data are not duplicated in this repository. Use the processed species-specific benchmark described in the manuscript and arrange it as documented in `docs/DATA_LAYOUT.md`.

For example:

```text
data/
└── Rattus_norvegicus/
    ├── train_Rattus_norvegicus_31.txt
    ├── valid_Rattus_norvegicus_31.txt
    └── test_Rattus_norvegicus_31.txt
```

## Running one species

Linux/macOS example:

```bash
export CENRES_SPECIES=Rattus_norvegicus
export CENRES_DATA_ROOT=/path/to/data
export CENRES_OUTPUT_ROOT=/path/to/outputs
export CENRES_GPU_ID=0

python cenres_ace_hyperparameter_search.py
```

Windows PowerShell example:

```powershell
$env:CENRES_SPECIES="Rattus_norvegicus"
$env:CENRES_DATA_ROOT="D:\path\to\data"
$env:CENRES_OUTPUT_ROOT="D:\path\to\outputs"
$env:CENRES_GPU_ID="0"

python cenres_ace_hyperparameter_search.py
```

Supported values of `CENRES_SPECIES` are:

- `Rattus_norvegicus`
- `Schistosoma_japonicum`
- `Saccharomyces_cerevisiae`
- `Mus_musculus`
- `Escherichia_coli`
- `Bacillus_velezensis`
- `Plasmodium_falciparum`
- `Oryza_sativa`
- `Arabidopsis_thaliana`

An optional environment variable `CENRES_MAX_TRIALS` can override the default random-search budget.

## Model selection and evaluation

For each hyperparameter trial, the best epoch is selected primarily by validation AUROC, with validation AUPRC used to break ties. Hyperparameter ranking is based on validation data only.

For manuscript-compatible fixed-specificity reporting, the main test metrics are evaluated at the operating point corresponding to `Sp = 0.900` on the test ROC curve. The script additionally reports results obtained by applying the validation-derived threshold to the test set.

The random seed for model training is fixed at 42. Deterministic PyTorch/CUDA settings are enabled where supported.

## Final manuscript configurations

The species-specific settings selected for the final CenRes-Ace models are stored in:

```text
configs/final_species_configs.json
```

These configurations include the BiGRU hidden size, Gaussian center-neighborhood radius, classifier size, optimizer settings, dropout rates, EMA decay, early-stopping patience, and selected epoch reported in the manuscript and supplementary material.

## Notes on reproducibility

- The public script removes machine-specific absolute paths used during development.
- Data and output locations are controlled with environment variables.
- The same model implementation is used for all nine species.
- Dataset redistribution is intentionally separated from source-code release; use the benchmark source described in the manuscript.

## Citation

Citation information will be updated after publication.
