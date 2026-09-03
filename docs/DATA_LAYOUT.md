# Data layout

CenRes-Ace uses the processed species-specific lysine acetylation benchmark described in the manuscript.

The repository does **not** duplicate the benchmark datasets. Place the processed files under a local data root with one folder per species.

Example:

```text
data/
└── Rattus_norvegicus/
    ├── train_Rattus_norvegicus_31.txt
    ├── valid_Rattus_norvegicus_31.txt
    └── test_Rattus_norvegicus_31.txt
```

The script also accepts `val_<species>_31.txt` or `dataset_<species>_31.txt` as the validation filename.

Supported species:

- `Rattus_norvegicus`
- `Schistosoma_japonicum`
- `Saccharomyces_cerevisiae`
- `Mus_musculus`
- `Escherichia_coli`
- `Bacillus_velezensis`
- `Plasmodium_falciparum`
- `Oryza_sativa`
- `Arabidopsis_thaliana`

Set the root folder through the environment variable `CENRES_DATA_ROOT`.
