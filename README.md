# CenRes-Ace

Official code and reproducibility repository for:

**CenRes-Ace: A Center-Aware Multi-Scale Residual CNN-BiGRU Framework for Species-Specific Lysine Acetylation-Site Prediction**

CenRes-Ace is a species-specific lysine acetylation-site prediction framework that integrates complementary residue-level descriptors, multi-scale residual convolution, channel and spatial attention, bidirectional contextual encoding, and Gaussian center-aware aggregation.

The repository provides the implementation, species-specific configurations, processed benchmark data used in the study, and documentation required to reproduce the reported analyses.

## Repository structure

```text
CenRes-Ace/
├── cenres_ace_hyperparameter_search.py
├── configs/
│   └── final_species_configs.json
├── data/
│   ├── data_train.zip
│   ├── data_valid.zip
│   └── data_test.zip
├── docs/
│   └── DATA_LAYOUT.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md

The same Python implementation is used for all nine species. Species-specific configurations selected for the final models reported in the manuscript are provided in:

configs/final_species_configs.json
Model architecture

CenRes-Ace uses four complementary residue-level feature groups:

21-dimensional one-hot residue identity encoding;
normalized relative position with respect to the candidate lysine;
20-dimensional BLOSUM62 substitution scores;
eight physicochemical descriptors.

The complete model consists of:

a 1 × 1 convolutional projection stem;
three multi-scale residual convolutional blocks;
standard convolution branches with kernel sizes 3 and 5;
a dilated convolution branch with kernel size 3 and dilation rate 2;
channel attention and spatial attention;
a bidirectional gated recurrent unit (BiGRU);
Gaussian center-aware aggregation;
a multilayer perceptron (MLP) classifier.

The input sequence window contains 31 residues centered on the candidate lysine, and each residue is represented by a 50-dimensional feature vector.

Experimental environment

The experiments reported in the manuscript were conducted on a server equipped with an NVIDIA GeForce RTX 4090 GPU.

The principal software environment was:

Python 3.8.20
PyTorch 2.4.1+cu121
PyTorch Geometric 2.6.1
scikit-learn 1.3.2
NumPy 1.24.3
pandas 2.0.3
Matplotlib 3.7.5
CUDA Toolkit 12.1

Exact Python package requirements are also provided in:

requirements.txt
Installation

Clone the repository:

git clone https://github.com/xingleyun/CenRes-Ace.git
cd CenRes-Ace

Install the required Python packages:

pip install -r requirements.txt

A CUDA-enabled GPU is recommended for model training and hyperparameter search.

Benchmark data

The repository contains the processed training, validation, and independent test partitions used in the study:

data/
├── data_train.zip
├── data_valid.zip
└── data_test.zip

These files correspond to the predefined species-specific benchmark partitions described in the manuscript.

The benchmark covers nine non-human species:

Rattus_norvegicus
Schistosoma_japonicum
Saccharomyces_cerevisiae
Mus_musculus
Escherichia_coli
Bacillus_velezensis
Plasmodium_falciparum
Oryza_sativa
Arabidopsis_thaliana

Detailed information on the expected directory structure and file naming is provided in:

docs/DATA_LAYOUT.md

After extraction, the data should be arranged according to the layout described in that document.

For example:

data/
└── Rattus_norvegicus/
    ├── train_Rattus_norvegicus_31.txt
    ├── valid_Rattus_norvegicus_31.txt
    └── test_Rattus_norvegicus_31.txt

The benchmark data were derived from the previously published species-specific lysine acetylation benchmark cited in the manuscript. The benchmark datasets are not covered by the MIT License applied to the CenRes-Ace source code and remain subject to the terms and attribution requirements of their original source.

Running CenRes-Ace

The same script supports all nine species. The target species and input/output locations are controlled through environment variables.

Linux/macOS example
export CENRES_SPECIES=Rattus_norvegicus
export CENRES_DATA_ROOT=/path/to/data
export CENRES_OUTPUT_ROOT=/path/to/outputs
export CENRES_GPU_ID=0

python cenres_ace_hyperparameter_search.py
Windows PowerShell example
$env:CENRES_SPECIES="Rattus_norvegicus"
$env:CENRES_DATA_ROOT="D:\path\to\data"
$env:CENRES_OUTPUT_ROOT="D:\path\to\outputs"
$env:CENRES_GPU_ID="0"

python cenres_ace_hyperparameter_search.py

Supported values of CENRES_SPECIES are:

Rattus_norvegicus
Schistosoma_japonicum
Saccharomyces_cerevisiae
Mus_musculus
Escherichia_coli
Bacillus_velezensis
Plasmodium_falciparum
Oryza_sativa
Arabidopsis_thaliana

The optional environment variable:

CENRES_MAX_TRIALS

can be used to override the default random-search budget.

Model training and selection

A separate model is trained for each species.

For each hyperparameter configuration:

model parameters are optimized using the predefined training set;
performance is evaluated on the predefined validation set after each epoch;
the checkpoint with the highest validation AUROC is retained;
validation AUPRC is used to resolve ties;
hyperparameter configurations are ranked using validation performance only.

The final species-specific configurations reported in the manuscript are stored in:

configs/final_species_configs.json

These configurations include:

batch size;
learning rate;
weight decay;
BiGRU hidden dimension;
Gaussian center-neighborhood radius;
MLP hidden dimension;
dropout rates;
label-smoothing coefficient;
exponential moving average decay;
early-stopping patience;
maximum number of epochs;
selected checkpoint epoch.
Evaluation protocol

The primary manuscript results follow the fixed-specificity benchmark protocol used in previous species-specific acetylation-site prediction studies.

For the primary threshold-dependent comparison, the operating point is selected from the test-set receiver operating characteristic curve to target:

Sp = 0.900

Sensitivity, accuracy, precision, F1 score, and Matthews correlation coefficient are then calculated at this operating point.

The script additionally reports:

AUROC;
AUPRC;
results obtained using a validation-derived threshold.

The test-set-derived fixed-specificity threshold is used only to reproduce the established benchmark reporting protocol and is not used for model training, hyperparameter optimization, or checkpoint selection.

Reproducibility

To improve reproducibility:

the random seed is fixed at 42 for Python, NumPy, and PyTorch;
deterministic PyTorch/CUDA operations are enabled where supported;
cuDNN benchmarking is disabled;
the same CenRes-Ace implementation is used for all nine species;
machine-specific absolute paths used during development have been removed;
data and output paths are controlled through environment variables;
the final species-specific configurations are publicly provided.

Because deep neural network training may still exhibit minor hardware- or library-dependent numerical variation, exact floating-point values can differ slightly across computing environments.

Baseline comparison

The repository contains the implementation of CenRes-Ace only.

The four previously published comparison methods:

PAIL
CapsNet
DeepDA-Ace
MDDeep-Ace

were not retrained or reimplemented in this study.

The Sn, Acc, Pre, and F1 values used for comparison in the manuscript were taken from the previously published benchmark evaluations described and cited in the manuscript.

Accordingly, the comparison standardizes the species-specific benchmark framework and the fixed-specificity operating point but does not imply that all competing methods were trained using identical procedures or optimization strategies.

License

The CenRes-Ace source code is released under the MIT License.

See:

LICENSE

for the full license text.

The MIT License applies to the CenRes-Ace source code developed for this repository. Benchmark datasets originating from previously published studies are not relicensed under the MIT License and remain subject to the terms and attribution requirements of their original sources.

Citation

If you use CenRes-Ace before publication of the accompanying article, please cite the repository as:

Xing L. CenRes-Ace. GitHub. 2026. https://github.com/xingleyun/CenRes-Ace

The citation information will be updated after publication of the accompanying manuscript.
