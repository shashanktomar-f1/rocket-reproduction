"""
reproduce_ucr.py - Reproduce ROCKET's main experiment on the UCR archive.

This script reproduces the primary experiment from Section 4.1 of the paper.
It runs ROCKET on all 128 UCR datasets using the default train/test split,
with 10000 kernels and 10 independent runs per dataset.

Reproduction decisions (documented for the report):
    1. We use our own rocket_implementation.py (similar algorithm but our own code)
    2. The original code uses RidgeClassifierCV(normalize=True), which is
       deprecated in modern scikit-learn. We use StandardScaler + RidgeClassifierCV
       as a functionally equivalent replacement.
    3. The original UCR script does NOT z-normalise time series before the
       ROCKET transform. We match this to stay faithful to what the authors
       actually ran.
    4. No random seed is set, matching the paper's nondeterministic approach
       Hence, the results will vary between runs but should be statistically similar.

Usage:
    # Full reproduction (all 128 datasets, 10 runs)
    python reproduce_ucr.py --data_path ../data/UCRArchive_2018 --output_path results --num_runs 10


    # With automatic comparison against the paper results
    python reproduce_ucr.py --data_path ../data/UCRArchive_2018 --output_path results \\
        --paper_results ../original_authors/results/results_ucr.csv

Authors: Shashank Sanjay Tomar & Manan Malik
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Import OUR implementation of ROCKET (not the author file)
from rocket_implementation import generate_kernels, apply_kernels

# Import our shared utility functions
from utils import (
    load_ucr_dataset,
    get_dataset_info,
    load_dataset_names,
    save_results,
    load_paper_results,
    compare_results,
    print_header,
    print_subheader,
    print_comparison_summary,
)


def build_classifier():
    """
    Build the classifier pipeline used for all UCR experiments

    The paper uses: RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)

    The normalize=True parameter was deprecated in scikit-learn 1.0 and removed
    in 1.2+. It internally applied feature standardisation before fitting.
    We replicate this behaviour using a StandardScaler in the pipeline.

    We set handle_unknown values by replacing NaN/inf that can arise when a
    feature has zero variance (constant across all training examples)

    Returns:
    
    classifier : sklearn Pipeline
        StandardScaler -> RidgeClassifierCV
    """
    return make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )

def run_single_dataset(dataset_name, data_path, num_runs, num_kernels):
    """
    Run the ROCKET experiment on one dataset for multiple runs

    Each run generates fresh random kernels, transforms the data, trains
    a ridge classifier, and evaluates accuracy. This matches the inner
    loop of the authors "reproduce_experiments_ucr.py"

    Parameters:
    
    dataset_name : str
        Name of the UCR dataset
    data_path : str
        Path to the parent directory containing dataset folders
    num_runs : int
        Number of independent runs (paper uses 10)
    num_kernels : int
        Number of random kernels (paper uses 10000)

    Returns:
    
    result : dict
        Contains accuracy_mean, accuracy_std, training time, test time,
        per-run accuracies, and the dataset info
    """
    # Load data
    X_train, Y_train, X_test, Y_test = load_ucr_dataset(dataset_name, data_path)
    info = get_dataset_info(X_train, Y_train, X_test, Y_test)

    # Storage for per-run results
    accuracies = np.zeros(num_runs)
    # 4 timing components: transform_train, transform_test, classifier_fit, classifier_predict
    timings = np.zeros((4, num_runs))

    for run in range(num_runs):

        # Generate random kernels
        input_length = X_train.shape[-1]
        kernels = generate_kernels(input_length, num_kernels)

        # Transform training data
        t_start = time.perf_counter()
        X_train_transform = apply_kernels(X_train, kernels)
        t_end = time.perf_counter()
        timings[0, run] = t_end - t_start

        # Transform test data
        t_start = time.perf_counter()
        X_test_transform = apply_kernels(X_test, kernels)
        t_end = time.perf_counter()
        timings[1, run] = t_end - t_start

        # Handle constant/degenerate features
        # Some kernels produce identical output for all training examples,
        # resulting in zero variance features. When the StandardScaler divides
        # by zero std, it produces inf/NaN. We replace these with 0,
        # which is safe because a constant feature carries no distinct
        # information anyway.
        X_train_transform = np.nan_to_num(
            X_train_transform, nan=0.0, posinf=0.0, neginf=0.0
        )
        X_test_transform = np.nan_to_num(
            X_test_transform, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Train classifier
        t_start = time.perf_counter()
        classifier = build_classifier()
        classifier.fit(X_train_transform, Y_train)
        t_end = time.perf_counter()
        timings[2, run] = t_end - t_start

        # Evaluate on test set
        t_start = time.perf_counter()
        accuracy = classifier.score(X_test_transform, Y_test)
        t_end = time.perf_counter()
        timings[3, run] = t_end - t_start

        accuracies[run] = accuracy

    # Aggregate results (matching the paper's reporting format)
    # Training time = mean(transform_train) + mean(classifier_fit)
    # Test time = mean(transform_test) + mean(classifier_predict)
    result = {
        "accuracy_mean": accuracies.mean(),
        "accuracy_standard_deviation": accuracies.std(),
        "time_training_seconds": timings.mean(axis=1)[[0, 2]].sum(),
        "time_test_seconds": timings.mean(axis=1)[[1, 3]].sum(),
        "all_accuracies": accuracies.tolist(),
        "dataset_info": info,
    }

    return result

def main():
    """Main function: parse arguments, run all datasets, save and compare."""

    parser = argparse.ArgumentParser(
        description="Reproduce ROCKET on the UCR archive (128 datasets)."
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Parent directory for UCR datasets.",
    )
    parser.add_argument(
        "--output_path", default="results",
        help="Directory for output CSVs (default: results).",
    )
    parser.add_argument(
        "--dataset_list", default=None,
        help="Text file of dataset names. If omitted, auto-detects from data_path",
    )
    parser.add_argument(
        "--num_runs", type=int, default=10,
        help="Runs per dataset (default: 10, matching the paper)",
    )
    parser.add_argument(
        "--num_kernels", type=int, default=10_000,
        help="Number of random kernels (default: 10000, matching the paper)",
    )
    parser.add_argument(
        "--paper_results", default=None,
        help="Path to the paper's results_ucr.csv for automatic comparison",
    )

    args = parser.parse_args()

    # Determine datasets
    if args.dataset_list:
        dataset_names = load_dataset_names(args.dataset_list)
    else:
        dataset_names = sorted([
            d for d in os.listdir(args.data_path)
            if os.path.isdir(os.path.join(args.data_path, d))
        ])

    if not dataset_names:
        print("ERROR: No datasets found. Check --data_path.")
        sys.exit(1)

    # Prepare results storage
    results = pd.DataFrame(
        index=dataset_names,
        columns=[
            "accuracy_mean",
            "accuracy_standard_deviation",
            "time_training_seconds",
            "time_test_seconds",
        ],
        data=0.0,
    )
    results.index.name = "dataset"

    detailed_results = {}

    # Run experiments
    print_header("ROCKET REPRODUCTION - UCR ARCHIVE")
    print(f"  Datasets:      {len(dataset_names)}")
    print(f"  Num kernels:   {args.num_kernels:,}")
    print(f"  Num runs:      {args.num_runs}")
    print(f"  Output path:   {args.output_path}")
    print()

    failed_datasets = []

    for i, dataset_name in enumerate(dataset_names):

        print_subheader(f"[{i + 1}/{len(dataset_names)}] {dataset_name}")

        try:
            result = run_single_dataset(
                dataset_name, args.data_path, args.num_runs, args.num_kernels,
            )

            results.loc[dataset_name, "accuracy_mean"] = result["accuracy_mean"]
            results.loc[dataset_name, "accuracy_standard_deviation"] = result["accuracy_standard_deviation"]
            results.loc[dataset_name, "time_training_seconds"] = result["time_training_seconds"]
            results.loc[dataset_name, "time_test_seconds"] = result["time_test_seconds"]

            detailed_results[dataset_name] = {
                "all_accuracies": result["all_accuracies"],
                "dataset_info": result["dataset_info"],
            }

            info = result["dataset_info"]
            print(
                f"  Train: {info['num_train']:>6}  |  Test: {info['num_test']:>6}  |  "
                f"Length: {info['ts_length']:>5}  |  Classes: {info['num_classes']:>3}"
            )
            print(
                f"  Accuracy: {result['accuracy_mean']:.6f} "
                f"(+/- {result['accuracy_standard_deviation']:.6f})  |  "
                f"Train: {result['time_training_seconds']:.2f}s  |  "
                f"Test: {result['time_test_seconds']:.2f}s"
            )

        except Exception as e:
            print(f"  FAILED: {e}")
            failed_datasets.append(dataset_name)

    # Save main results
    print_header("SAVING RESULTS")

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    save_results(results, os.path.join(output_dir, "our_results_ucr.csv"))

    # Save detailed per-run results
    detailed_rows = []
    for dname, detail in detailed_results.items():
        for run_idx, acc in enumerate(detail["all_accuracies"]):
            detailed_rows.append({
                "dataset": dname,
                "run": run_idx,
                "accuracy": acc,
            })
    if detailed_rows:
        detailed_df = pd.DataFrame(detailed_rows)
        save_results(detailed_df, os.path.join(output_dir, "our_results_ucr_detailed.csv"))

    # Report failures
    if failed_datasets:
        print(f"\nWARNING: {len(failed_datasets)} datasets failed:")
        for name in failed_datasets:
            print(f"  - {name}")

    # Compare against paper
    if args.paper_results and os.path.exists(args.paper_results):
        print_header("COMPARISON WITH PAPER RESULTS")
        paper_df = load_paper_results(args.paper_results)
        comparison = compare_results(results, paper_df)
        print_comparison_summary(comparison)
        save_results(comparison, os.path.join(output_dir, "comparison_ucr.csv"))

    print_header("EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
