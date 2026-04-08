"""
reproduce_scalability.py - Reproduce ROCKET's scalability experiments

Here we are reproducing two claims from Section 4.2 of the paper:

1. Training time scales linearly with time series LENGTH (Section 4.2.2)
   - Uses the InlineSkate dataset from UCR, truncated to different lengths.

2. Training time scales linearly with training set SIZE (Section 4.2.1)
   - The original used the Formosat-2 Satellite dataset (NOT publicly available).
   - We use a large UCR dataset as a substitute to verify the linear scaling
     claim with accessible data.
   - This limitation is documented in our report

Usage:
    python reproduce_scalability.py --data_path ../data/UCRArchive_2018 --output_path results --experiment all
    python reproduce_scalability.py --data_path ../data/UCRArchive_2018 --output_path results --experiment length
    python reproduce_scalability.py --data_path ../data/UCRArchive_2018 --output_path results --experiment size

Authors: Shashank Sanjay Tomar & Manan Malik
"""

import argparse
import os
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from rocket_implementation import generate_kernels, apply_kernels
from utils import (
    load_ucr_dataset,
    save_results,
    print_header,
    print_subheader,
)


def experiment_time_series_length(data_path, output_path, num_kernels=10_000, num_runs=5):
    """
    Paper's section 4.2.2: Scalability with time series length

    Loads InlineSkate, truncates to lengths [32, 64, 128, 256, 512, 1024, 2048]
    and measures training time at each length.

    The paper reports a linear relationship between length and training time
    """
    print_header("SCALABILITY: Time Series Length (InlineSkate)")

    dataset_name = "InlineSkate"
    X_train, Y_train, X_test, Y_test = load_ucr_dataset(dataset_name, data_path)
    full_length = X_train.shape[1]

    print(f"  Dataset:     {dataset_name}")
    print(f"  Full length: {full_length}")
    print(f"  Train size:  {X_train.shape[0]}")
    print(f"  Test size:   {X_test.shape[0]}")

    # Test these lengths (powers of 2, up to the dataset's full length)
    target_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    lengths = [l for l in target_lengths if l <= full_length]

    results_rows = []

    for length in lengths:
        print_subheader(f"Length = {length}")

        run_times = np.zeros(num_runs)

        for run in range(num_runs):
            # Truncate time series to this length
            X_tr = X_train[:, :length]
            X_te = X_test[:, :length]

            # Time the full pipeline: generate kernels + transform + fit
            t_start = time.perf_counter()

            kernels = generate_kernels(length, num_kernels)
            X_tr_transform = apply_kernels(X_tr, kernels)
            X_te_transform = apply_kernels(X_te, kernels)

            classifier = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            )
            classifier.fit(X_tr_transform, Y_train)

            t_end = time.perf_counter()
            run_times[run] = t_end - t_start

        mean_time = run_times.mean()
        results_rows.append({
            "length": length,
            "time_training_seconds": round(mean_time, 2),
        })
        print(f"  Mean training time: {mean_time:.2f}s")

    results_df = pd.DataFrame(results_rows)
    save_results(results_df, os.path.join(output_path, "our_results_scalability_ts_length.csv"))
    return results_df


def experiment_training_set_size(data_path, output_path, num_kernels=10_000, num_runs=5):
    """
    Paper's section 4.2.1 (partial): Scalability with training set size

    The original experiment used the Formosat-2 Satellite dataset, which is
    NOT publicly available. We use a large UCR dataset instead and subsample
    at increasing sizes to test the linear scaling claim by the authors

    This is a legitimate substitute as the claim being tested is about
    computational scaling, not about a specific dataset's accuracy.
    """
    print_header("SCALABILITY: Training Set Size (Substitute Dataset)")

    # Try to find a sufficiently large dataset
    candidate_datasets = ["StarLightCurves", "ElectricDevices", "Wafer", "ECG5000"]
    dataset_name = None

    for name in candidate_datasets:
        try:
            X_train, Y_train, X_test, Y_test = load_ucr_dataset(name, data_path)
            if X_train.shape[0] >= 500:
                dataset_name = name
                break
        except FileNotFoundError:
            continue

    if dataset_name is None:
        print("ERROR: No suitable large dataset found.")
        return None

    print(f"  Dataset:         {dataset_name}")
    print(f"  Full train size: {X_train.shape[0]}")
    print(f"  TS length:       {X_train.shape[1]}")
    print(f"  NOTE: Original paper used Formosat-2 Satellite dataset")
    print(f"        (not publicly available). This is a substitute.")

    # Subsample sizes (powers of 2, up to the full training set)
    max_power = int(np.floor(np.log2(X_train.shape[0])))
    min_power = max(4, max_power - 6)  # At least 16 examples
    sizes = [2 ** p for p in range(min_power, max_power + 1)]

    results_rows = []

    for size in sizes:
        print_subheader(f"Training size = {size}")

        run_times = np.zeros(num_runs)
        run_accs = np.zeros(num_runs)

        for run in range(num_runs):
            # Randomly subsample the training data
            indices = np.random.choice(X_train.shape[0], size=size, replace=False)
            X_sub = X_train[indices]
            Y_sub = Y_train[indices]

            # Time the full pipeline
            t_start = time.perf_counter()

            kernels = generate_kernels(X_sub.shape[1], num_kernels)
            X_sub_transform = apply_kernels(X_sub, kernels)
            X_te_transform = apply_kernels(X_test, kernels)

            classifier = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            )
            classifier.fit(X_sub_transform, Y_sub)

            t_end = time.perf_counter()

            run_times[run] = t_end - t_start
            run_accs[run] = classifier.score(X_te_transform, Y_test)

        results_rows.append({
            "num_training_examples": size,
            "accuracy_mean": round(run_accs.mean(), 6),
            "accuracy_std": round(run_accs.std(), 6),
            "time_training_seconds": round(run_times.mean(), 2),
        })
        print(f"  Time: {run_times.mean():.2f}s  |  Accuracy: {run_accs.mean():.4f}")

    results_df = pd.DataFrame(results_rows)
    save_results(results_df, os.path.join(output_path, "our_results_scalability_train_size.csv"))
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce ROCKET scalability experiments."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="results")
    parser.add_argument(
        "--experiment", default="all", choices=["all", "length", "size"],
    )
    parser.add_argument("--num_kernels", type=int, default=10_000)
    parser.add_argument("--num_runs", type=int, default=5)

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    if args.experiment in ("all", "length"):
        experiment_time_series_length(
            args.data_path, args.output_path, args.num_kernels, args.num_runs,
        )

    if args.experiment in ("all", "size"):
        experiment_training_set_size(
            args.data_path, args.output_path, args.num_kernels, args.num_runs,
        )

    print_header("SCALABILITY EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
