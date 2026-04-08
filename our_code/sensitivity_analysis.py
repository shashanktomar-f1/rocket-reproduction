"""
sensitivity_analysis.py - Reproduce and extend ROCKET's sensitivity analysis

Tests the claims from Section 4.3 of the paper:
    4.3.1: Effect of number of kernels (100 to 10000)
    4.3.3: Effect of bias (with vs without)
    4.3.4: Effect of dilation (with vs without)
    4.3.5: Effect of padding (random vs never vs always)
    4.3.6: Effect of features (PPV only vs max only vs both)

For each experiment, we change ONE aspect of the ROCKET pipeline while keeping
everything else identical, and measure the effect on accuracy. It is essentialy an
"ablation study" where you remove or change one component to see how much
it contributes.



Usage:
    python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment all
    python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment features
    python sensitivity_analysis.py --data_path ../data/UCRArchive_2018 --output_path results --experiment num_kernels

Authors: Shashank Sanjay Tomar & Manan Malik
"""

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from numba import njit, prange

# Import our standard ROCKET implementation
from rocket_implementation import generate_kernels, apply_kernels, apply_single_kernel
from utils import (
    load_ucr_dataset,
    load_dataset_names,
    save_results,
    print_header,
    print_subheader,
)



# MODIFIED FEATURE EXTRACTION (for PPV-only and max-only experiments)

# These are variants of apply_kernels that extract only ONE feature per kernel
# instead of both. This lets us test the paper's claim that PPV is the most
# critical feature

@njit(parallel=True, fastmath=True)
def apply_kernels_ppv_only(X, kernels):
    """Extract ONLY the PPV feature from each kernel (1 feature per kernel)"""
    weights, lengths, biases, dilations, paddings = kernels
    num_examples, _ = X.shape
    num_kernels = len(lengths)
    features = np.zeros((num_examples, num_kernels), dtype=np.float64)

    for i in prange(num_examples):
        weight_idx = 0
        for j in range(num_kernels):
            w_end = weight_idx + lengths[j]
            ppv, _ = apply_single_kernel(
                X[i], weights[weight_idx:w_end],
                lengths[j], biases[j], dilations[j], paddings[j],
            )
            features[i, j] = ppv
            weight_idx = w_end
    return features


@njit(parallel=True, fastmath=True)
def apply_kernels_max_only(X, kernels):
    """Extract ONLY the max feature from each kernel (1 feature per kernel)"""
    weights, lengths, biases, dilations, paddings = kernels
    num_examples, _ = X.shape
    num_kernels = len(lengths)
    features = np.zeros((num_examples, num_kernels), dtype=np.float64)

    for i in prange(num_examples):
        weight_idx = 0
        for j in range(num_kernels):
            w_end = weight_idx + lengths[j]
            _, max_val = apply_single_kernel(
                X[i], weights[weight_idx:w_end],
                lengths[j], biases[j], dilations[j], paddings[j],
            )
            features[i, j] = max_val
            weight_idx = w_end
    return features



# MODIFIED KERNEL GENERATORS (for dilation, bias, padding ablation)

# Each of these generates kernels identically to the standard version,
# except for ONE parameter that is fixed rather than random

@njit("Tuple((float64[:], int32[:], float64[:], int32[:], int32[:]))(int64, int64)")
def generate_kernels_no_dilation(input_length, num_kernels):
    """Kernels with dilation always = 1 (no multi-scale patterns)"""
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)
    w_idx = 0
    for i in range(num_kernels):
        _len = lengths[i]
        _w = np.random.normal(0, 1, _len)
        weights[w_idx:w_idx + _len] = _w - _w.mean()
        biases[i] = np.random.uniform(-1, 1)
        dilations[i] = np.int32(1)  # FIXED: no dilation
        paddings[i] = ((_len - 1) * 1) // 2 if np.random.randint(2) == 1 else 0
        w_idx += _len
    return weights, lengths, biases, dilations, paddings


@njit("Tuple((float64[:], int32[:], float64[:], int32[:], int32[:]))(int64, int64)")
def generate_kernels_no_bias(input_length, num_kernels):
    """Kernels with bias always = 0"""
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)
    w_idx = 0
    for i in range(num_kernels):
        _len = lengths[i]
        _w = np.random.normal(0, 1, _len)
        weights[w_idx:w_idx + _len] = _w - _w.mean()
        biases[i] = np.float64(0.0)  # FIXED: no bias
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_len - 1)))
        dilations[i] = np.int32(dilation)
        paddings[i] = ((_len - 1) * np.int32(dilation)) // 2 if np.random.randint(2) == 1 else 0
        w_idx += _len
    return weights, lengths, biases, dilations, paddings


@njit("Tuple((float64[:], int32[:], float64[:], int32[:], int32[:]))(int64, int64)")
def generate_kernels_no_padding(input_length, num_kernels):
    """Kernels with padding always = 0 (never pad)"""
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)
    w_idx = 0
    for i in range(num_kernels):
        _len = lengths[i]
        _w = np.random.normal(0, 1, _len)
        weights[w_idx:w_idx + _len] = _w - _w.mean()
        biases[i] = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_len - 1)))
        dilations[i] = np.int32(dilation)
        paddings[i] = np.int32(0)  # FIXED: never pad
        w_idx += _len
    return weights, lengths, biases, dilations, paddings


@njit("Tuple((float64[:], int32[:], float64[:], int32[:], int32[:]))(int64, int64)")
def generate_kernels_always_pad(input_length, num_kernels):
    """Kernels with padding always applied"""
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)
    w_idx = 0
    for i in range(num_kernels):
        _len = lengths[i]
        _w = np.random.normal(0, 1, _len)
        weights[w_idx:w_idx + _len] = _w - _w.mean()
        biases[i] = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_len - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation
        paddings[i] = ((_len - 1) * dilation) // 2  # FIXED: always pad
        w_idx += _len
    return weights, lengths, biases, dilations, paddings



# GENERIC EXPERIMENT RUNNER


def run_variant(X_train, Y_train, X_test, Y_test,
                num_kernels=10_000, num_runs=5,
                kernel_gen=generate_kernels,
                kernel_apply=apply_kernels):
    """
    Run a ROCKET variant and return mean accuracy and std

    By swapping kernel_gen or kernel_apply, we can test any ablation
    """
    accuracies = np.zeros(num_runs)
    input_length = X_train.shape[-1]

    for run in range(num_runs):
        kernels = kernel_gen(input_length, num_kernels)
        X_tr_t = kernel_apply(X_train, kernels)
        X_te_t = kernel_apply(X_test, kernels)
        X_tr_t = np.nan_to_num(X_tr_t, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_t = np.nan_to_num(X_te_t, nan=0.0, posinf=0.0, neginf=0.0)

        clf = make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        clf.fit(X_tr_t, Y_train)
        accuracies[run] = clf.score(X_te_t, Y_test)

    return accuracies.mean(), accuracies.std()



# INDIVIDUAL EXPERIMENTS


def experiment_num_kernels(dataset_names, data_path, output_path, num_runs=5):
    """How does accuracy change with the number of kernels"""
    print_header("SENSITIVITY: Number of Kernels")
    kernel_counts = [100, 500, 1_000, 2_500, 5_000, 10_000]
    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)

        for k in kernel_counts:
            mean_acc, std_acc = run_variant(X_tr, Y_tr, X_te, Y_te, num_kernels=k, num_runs=num_runs)
            rows.append({"dataset": dname, "num_kernels": k, "accuracy_mean": mean_acc, "accuracy_std": std_acc})
            print(f"  k={k:>6,}: {mean_acc:.4f} (+/- {std_acc:.4f})")

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "sensitivity_num_kernels.csv"))
    return df


def experiment_features(dataset_names, data_path, output_path, num_runs=5):
    """PPV only vs max only vs both"""
    print_header("SENSITIVITY: Feature Ablation (PPV vs Max vs Both)")
    configs = [
        ("both (default)", apply_kernels),
        ("ppv_only", apply_kernels_ppv_only),
        ("max_only", apply_kernels_max_only),
    ]
    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)

        for feat_name, applier in configs:
            mean_acc, std_acc = run_variant(X_tr, Y_tr, X_te, Y_te, num_runs=num_runs, kernel_apply=applier)
            rows.append({"dataset": dname, "feature_type": feat_name, "accuracy_mean": mean_acc, "accuracy_std": std_acc})
            print(f"  {feat_name:>16}: {mean_acc:.4f} (+/- {std_acc:.4f})")

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "sensitivity_features.csv"))
    return df


def experiment_dilation(dataset_names, data_path, output_path, num_runs=5):
    """With dilation vs without"""
    print_header("SENSITIVITY: Effect of Dilation")
    configs = [
        ("with_dilation (default)", generate_kernels),
        ("no_dilation", generate_kernels_no_dilation),
    ]
    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)

        for var_name, gen_func in configs:
            mean_acc, std_acc = run_variant(X_tr, Y_tr, X_te, Y_te, num_runs=num_runs, kernel_gen=gen_func)
            rows.append({"dataset": dname, "variant": var_name, "accuracy_mean": mean_acc, "accuracy_std": std_acc})
            print(f"  {var_name:>25}: {mean_acc:.4f} (+/- {std_acc:.4f})")

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "sensitivity_dilation.csv"))
    return df


def experiment_bias(dataset_names, data_path, output_path, num_runs=5):
    """With bias vs without"""
    print_header("SENSITIVITY: Effect of Bias")
    configs = [
        ("with_bias (default)", generate_kernels),
        ("no_bias", generate_kernels_no_bias),
    ]
    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)

        for var_name, gen_func in configs:
            mean_acc, std_acc = run_variant(X_tr, Y_tr, X_te, Y_te, num_runs=num_runs, kernel_gen=gen_func)
            rows.append({"dataset": dname, "variant": var_name, "accuracy_mean": mean_acc, "accuracy_std": std_acc})
            print(f"  {var_name:>20}: {mean_acc:.4f} (+/- {std_acc:.4f})")

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "sensitivity_bias.csv"))
    return df


def experiment_padding(dataset_names, data_path, output_path, num_runs=5):
    """Random padding vs never pad vs always pad"""
    print_header("SENSITIVITY: Effect of Padding")
    configs = [
        ("random_padding (default)", generate_kernels),
        ("no_padding", generate_kernels_no_padding),
        ("always_padding", generate_kernels_always_pad),
    ]
    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)

        for var_name, gen_func in configs:
            mean_acc, std_acc = run_variant(X_tr, Y_tr, X_te, Y_te, num_runs=num_runs, kernel_gen=gen_func)
            rows.append({"dataset": dname, "variant": var_name, "accuracy_mean": mean_acc, "accuracy_std": std_acc})
            print(f"  {var_name:>25}: {mean_acc:.4f} (+/- {std_acc:.4f})")

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "sensitivity_padding.csv"))
    return df



# DEFAULT DATASET SUBSET (if you want a quick run - only 20 datasets)


DEFAULT_DATASETS = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "CBF",
    "ChlorineConcentration", "Coffee", "ECG200", "ECG5000",
    "FaceFour", "GunPoint", "Ham", "Herring", "ItalyPowerDemand",
    "Meat", "OSULeaf", "Plane", "StarLightCurves",
    "Strawberry", "Wafer",
]



# MAIN


def main():
    parser = argparse.ArgumentParser(description="ROCKET sensitivity analysis")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="results")
    parser.add_argument("--dataset_list", default=None)
    parser.add_argument(
        "--experiment", default="all",
        choices=["all", "num_kernels", "features", "dilation", "bias", "padding"],
    )
    parser.add_argument("--num_runs", type=int, default=5)

    args = parser.parse_args()

    if args.dataset_list:
        dataset_names = load_dataset_names(args.dataset_list)
    else:
        # Auto-detect all datasets from the data directory
        dataset_names = sorted([
            d for d in os.listdir(args.data_path)
            if os.path.isdir(os.path.join(args.data_path, d))
        ])
        print(f"Auto-detected {len(dataset_names)} datasets.")

    os.makedirs(args.output_path, exist_ok=True)
    print(f"\n  Datasets:    {len(dataset_names)}")
    print(f"  Num runs:    {args.num_runs}")
    print(f"  Experiment:  {args.experiment}")
    print()

    experiments = {
        "num_kernels": experiment_num_kernels,
        "features": experiment_features,
        "dilation": experiment_dilation,
        "bias": experiment_bias,
        "padding": experiment_padding,
    }

    if args.experiment == "all":
        for name, func in experiments.items():
            func(dataset_names, args.data_path, args.output_path, args.num_runs)
    else:
        experiments[args.experiment](dataset_names, args.data_path, args.output_path, args.num_runs)

    print_header("ALL SENSITIVITY EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
