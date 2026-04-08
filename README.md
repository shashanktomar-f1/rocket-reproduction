# ROCKET Reproduction and Improvement

**COMP41850 - AI for Time Series | University College Dublin**

## Authors

- Shashank Sanjay Tomar
- Manan Malik

## Introduct to this Project:

This project reproduces and attempts to improve the results from the ROCKET paper:

> Dempster, A., Petitjean, F., & Webb, G. I. (2020). **ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.** *Data Mining and Knowledge Discovery*, 34(5), 1454–1495.  

To summarise, ROCKET takes a time series, throws 10000 random convolutional kernels at it, extracts 2 features per kernel (proportion of positive values and global max), and feeds the resulting 20000-feature vector into a simple ridge regression classifier. It's shockingly fast and surprisingly accurate and is on par with methods that take days to train.

We reproduced the main accuracy results across all 128 UCR archive datasets, the scalability experiments, the full sensitivity analysis, and then proposed and tested 5 improvements of our own.

## Repository Structure

```
OUR REPO/
│
├── configs/                                    # Dataset list files
│   ├── bakeoff_85.txt                          # Names of the 85 original UCR bake-off datasets
│   └── test_datasets.txt                       # 3 small datasets for a quick sanity testing
│
├── data/                                       # UCR archive datasets (not in git, please refer to the setup section below)
│   └── UCRArchive_2018/
│
├── original_authors/                           # Authors' original code and results - UNTOUCHED
│   ├── Authors_README.md                       # Authors' original README
│   ├── README.md                               # Our note explaining what this folder contains
│   ├── rocket_functions.py                     # Their ROCKET implementation
│   ├── reproduce_experiments_ucr.py            # Their UCR experiment script
│   ├── reproduce_experiments_scalability.py    # Their scalability script
│   └── results/                                # Their published result CSVs
│       ├── results_ucr.csv                     # Accuracy on 128 datasets (10 runs)
│       ├── results_ucr_resamples.csv           # Accuracy with resampled splits
│       ├── results_scalability_training_set_size.csv
│       └── results_scalability_time_series_length.csv
│
├── our_code/                                   # Everything we wrote
│   ├── rocket_implementation.py                # Our own rewrite of the ROCKET algorithm
│   ├── utils.py                                # Shared helper functions: data loading, evaluation, timing
│   ├── reproduce_ucr.py                        # Main reproduction (128 datasets, 10 runs each)
│   ├── reproduce_scalability.py                # Reproduction of scalability by TS length and training set size
│   ├── sensitivity_analysis.py                 # Ablation studies: kernels, PPV/max, dilation, bias, padding to test the claims by the authors in the original paper
│   ├── improvements.py                         # Our 5 proposed improvements
│   ├── generate_figures.py                     # Generates all report figures from result CSVs
│   ├── results/                                # All output CSVs from our experiments (all ours)
│   │   ├── our_results_ucr.csv                 # Our reproduced accuracy (128 datasets)
│   │   ├── our_results_ucr_detailed.csv        # Per-run accuracy for every dataset
│   │   ├── comparison_ucr.csv                  # Side-by-side comparison of our results vs paper
│   │   ├── our_results_scalability_ts_length.csv
│   │   ├── our_results_scalability_train_size.csv
│   │   ├── sensitivity_num_kernels.csv
│   │   ├── sensitivity_features.csv
│   │   ├── sensitivity_dilation.csv
│   │   ├── sensitivity_bias.csv
│   │   ├── sensitivity_padding.csv
│   │   ├── improvement_classifiers.csv
│   │   ├── improvement_classifiers_summary.csv
│   │   ├── improvement_normalisation.csv
│   │   ├── improvement_normalisation_summary.csv
│   │   ├── improvement_stability.csv
│   │   ├── improvement_stability_summary.csv
│   │   ├── improvement_feature_selection.csv
│   │   ├── improvement_feature_selection_summary.csv
│   │   ├── improvement_ensemble.csv
│   │   └── improvement_ensemble_summary.csv
│   └── figures/                                # All report figures
│       ├── fig1_accuracy_comparison.png
│       ├── fig2_scalability_length.png
│       ├── fig3_num_kernels.png
│       ├── fig4_feature_ablation.png
│       ├── fig5_component_ablation.png
│       ├── fig6_classifiers.png
│       ├── fig7_normalisation.png
│       ├── fig8_stability.png
│       ├── fig9_feature_selection.png
│       └── fig10_ensemble.png
│                                     
├── requirements.txt                            # Python dependencies
└── README.md                                   # You're reading this
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/shashanktomar-f1/rocket-reproduction.git
cd rocket-reproduction
```

### 2. Set up Python Libraries

We used Python 3.14 on windows but any Python 3.8+ should work.

```bash
python -m venv venv
venv\Scripts\activate     
pip install -r requirements.txt
```

### 3. Download the UCR archive

The datasets are too large to include in the repo. You need to download them yourself:

1. Go to [https://www.cs.ucr.edu/~eamonn/time_series_data_2018/](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
2. Read the briefing PDF as it contains the password you need to download the archive
3. Download and unzip into the `data/` folder

Your data folder should look like this:
```
data/UCRArchive_2018/
├── ACSF1/
│   ├── ACSF1_TRAIN.tsv
│   └── ACSF1_TEST.tsv
├── Adiac/
│   ├── Adiac_TRAIN.tsv
│   └── Adiac_TEST.tsv
└── ... (128 dataset folders)
```

**Note:** The archive may get downloaded as `.tsv` files rather than `.txt` files depending on which version you download. Our code handles both formats automatically, this was one of the first issues we ran into during reproduction.

## Running the Experiments

All scripts live in `our_code/`. Run everything from that directory:

```bash
cd our_code
```

### Quick sanity test (begin with this)

Before running the full experiments, make sure everything works on 3 small datasets:

```bash
python reproduce_ucr.py --data_path ../data/UCRArchive_2018 --output_path results --num_runs 3 --dataset_list ../configs/test_datasets.txt
```

This should finish in under a minute. If Coffee, GunPoint, and ItalyPowerDemand all show results without errors, you're good to go.

### Experiment 1: Main UCR reproduction (128 datasets)

This is the big one as it reproduces the paper's primary accuracy results. It may take about 2-3 hours depending on your hardware.

```bash
python reproduce_ucr.py --data_path ../data/UCRArchive_2018 --output_path results --num_runs 10 --paper_results ../original_authors/results/results_ucr.csv
```

The `--paper_results` flag automatically prints a comparison summary at the end.

### Experiment 2: Scalability

This one tests whether training time scales linearly with time series length (using InlineSkate) and training set size (using StarLightCurves as a substitute as the paper's original Formosat-2 Satellite dataset is not publicly available).

```bash
python reproduce_scalability.py --data_path ../data/UCRArchive_2018 --output_path results --experiment all
```

### Experiment 3: Sensitivity analysis

Reproduces all five ablation studies from Section 4.3 of the paper on the 85 bake-off datasets, 10 runs each.

```bash
# All five experiments at once
python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment all --num_runs 10 --dataset_list ../configs/bakeoff_85.txt

# Or run them individually:
python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment num_kernels --num_runs 10 --dataset_list ../configs/bakeoff_85.txt
python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment features --num_runs 10 --dataset_list ../configs/bakeoff_85.txt
python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment dilation --num_runs 10 --dataset_list ../configs/bakeoff_85.txt
python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment bias --num_runs 10 --dataset_list ../configs/bakeoff_85.txt
python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment padding --num_runs 10 --dataset_list ../configs/bakeoff_85.txt
```

### Experiment 4: Our proposed improvements

Evaluates five proposed improvements: alternative classifiers, z-normalisation, seed stability (30 runs), feature selection, and ensemble of smaller models.

```bash
# All five improvements at once
python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment all

# Or individually:
python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment classifiers
python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment normalisation
python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment stability
python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment feature_selection
python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment ensemble
```

### Generating figures

Once all experiments have been run and the result CSVs are in `results/`, generate the report figures:

```bash
python generate_figures.py --results_path results --paper_results_path ../original_authors/results --output_path figures
```

This produces all 10 figures as PNG files in `our_code/figures/`

## Config Files

| File | What it is |
|------|------------|
| `configs/test_datasets.txt` | 3 small datasets (Coffee, GunPoint, ItalyPowerDemand) used for quick sanity checks before launching full experiments |
| `configs/bakeoff_85.txt` | The 85 original UCR bake-off datasets, used for sensitivity analysis to match the paper's protocol |

## Issues We Encountered

These are things that tripped us up during reproduction and are worth knowing about:

- **`.tsv` vs `.txt` files:** The UCR archive we downloaded had `.tsv` extensions, but the authors' code expects `.txt`. Our code handles both the extensions automatically.
- **`np.NINF` removed in NumPy 2.0:** The authors' code uses `np.NINF` which was removed in NumPy 2.0. Hence, we use `-np.inf` instead.
- **`normalize=True` removed in scikit-learn 1.2:** The authors use `RidgeClassifierCV(normalize=True)` which no longer exists. We use a `StandardScaler + RidgeClassifierCV` pipeline as an equivalent replacement.
- **Infinity values from constant features:** Some datasets produce features with zero variance, causing `StandardScaler` to output infinity. We handle this with `np.nan_to_num()` after the ROCKET transform.
- **LinearSVC convergence warnings:** LinearSVC sometimes fails to converge within the default iteration limit on ROCKET-transformed features. This doesn't crash the experiment but produces warnings.
- **Formosat-2 dataset not available publicly:** The scalability-by-training-size experiment in the paper uses a private satellite dataset. We use StarLightCurves from the UCR archive as a substitute.

## Reproducibility Notes

- ROCKET is nondeterministic and no random seed is set, matching the paper's approach. Your results will differ slightly from ours but should be statistically similar.
- We ran all experiments on a single machine. Results are in `our_code/results/` if you want to skip re-running and go straight to analysis.
- The `generate_figures.py` script reads from the result CSVs, so you can regenerate figures without re-running experiments.

## Hardware

- **CPU:** AMD Ryzen 9 9900X
- **RAM:** 32 GB DDR5
- **GPU:** NVIDIA RTX 5070 (not used as ROCKET runs entirely on CPU via Numba)
- **OS:** Windows



## References

- Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery*, 34(5), 1454–1495.
- Original Paper Authors' original code: [https://github.com/angus924/rocket](https://github.com/angus924/rocket)
- UCR Time Series Archive: [https://www.cs.ucr.edu/~eamonn/time_series_data_2018/](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
