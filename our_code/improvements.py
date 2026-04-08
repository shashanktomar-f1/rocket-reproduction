"""
improvements.py - Our proposed improvements to ROCKET and their evaluation

We propose and test five improvements:

Improvement 1: Alternative classifiers
    The paper uses ridge regression and we test whether other classifiers
    (Logistic Regression, Linear SVM, Random Forest) perform better on
    the ROCKET transformed features 

Improvement 2: Z-normalisation before the ROCKET transform
    the paper's UCR code does NOT normalise time series, but their own
    GitHub repo documentation says to normalise. We test whether adding normalisation
    helps, hurts, or makes no difference

Improvement 3: Seed stability analysis (30 runs)
    The paper reports std from 10 runs. We extend to 30 runs to better
    characterise the variance and test whether ROCKET is as stable as
    the paper suggests or not

Improvement 4: Feature selection (removing low-variance features)
    ROCKET produces 20000 features, but many of them may be uninformative
    the paper mentions feature selection as a future work (Section 5)
    We test whether removing low-variance features improves accuracy
    or speeds up classifier training.

Improvement 5: Ensemble of smaller ROCKET models
    Instead of having one ROCKET with 10000 kernels, we test whether combining
    multiple smaller ROCKET runs (for ex: 3 x 5000 kernels) by majority
    voting produces better or a more stable accuracy

Usage:
    python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment all
    python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment classifiers
    python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment normalisation
    python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment stability
    python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment feature_selection
    python improvements.py --data_path ../data/UCRArchive_2018 --output_path results --experiment ensemble

Authors: Shashank Sanjay Tomar & Manan Malik
"""

import argparse
import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold

from rocket_implementation import generate_kernels, apply_kernels
from utils import (
    load_ucr_dataset,
    load_dataset_names,
    z_normalise,
    save_results,
    print_header,
    print_subheader,
    Timer,
)



# DEFAULT DATASET SUBSET (if you want a quick run - only 20 datasets)

DEFAULT_DATASETS = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "CBF",
    "ChlorineConcentration", "Coffee", "ECG200", "ECG5000",
    "FaceFour", "GunPoint", "Ham", "Herring", "ItalyPowerDemand",
    "Meat", "OSULeaf", "Plane", "StarLightCurves",
    "Strawberry", "Wafer",
]



# IMPROVEMENT 1: ALTERNATIVE CLASSIFIERS


def get_classifiers():
    """
    Return the classifiers we want to compare

    Each one is wrapped in a pipeline with StandardScaler for a fair comparison.
    The paper's baseline setup used RidgeClassifierCV with internal normalisation,
    so all classifiers get the same preprocessing.

    Chose these classifiers because:
    - RidgeCV: the paper's baseline - linear, fast, built-in cross-validation
    - LogisticRegression: another linear model, uses gradient-based optimisation
    - LinearSVC: linear SVM, different loss function (hinge vs squared error)
    - RandomForest: non-linear, can capture feature interactions
    """
    return {
        "RidgeCV (baseline)": make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", 
            ),
        ),
        "LinearSVC": make_pipeline(
            StandardScaler(),
            LinearSVC(max_iter=5000, dual=False),
        ),
        "RandomForest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1,
            ),
        ),
    }


def experiment_classifiers(dataset_names, data_path, output_path, num_runs=5):
    """
    Improvement 1: Compare classifiers on ROCKET transformed features

    Hypothesis: A different classifier might exploit 20000 ROCKET features
    better than ridge regression on some datasets

    For fair comparison, for each run we generate kernels ONCE and apply them,
    then train each classifier on the SAME transformed data. This isolates
    the classifier as the only variable.
    """
    print_header("IMPROVEMENT 1: Alternative Classifiers")

    classifiers = get_classifiers()
    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)

        for run in range(num_runs):
            # Generate and apply kernels ONCE (shared across classifiers)
            kernels = generate_kernels(X_tr.shape[-1], 10_000)
            X_tr_t = apply_kernels(X_tr, kernels)
            X_te_t = apply_kernels(X_te, kernels)

            # Clean any inf/nan from degenerate features
            X_tr_t = np.nan_to_num(X_tr_t, nan=0.0, posinf=0.0, neginf=0.0)
            X_te_t = np.nan_to_num(X_te_t, nan=0.0, posinf=0.0, neginf=0.0)

            for clf_name, clf in classifiers.items():
                try:
                    with Timer() as t:
                        clf.fit(X_tr_t, Y_tr)
                    acc = clf.score(X_te_t, Y_te)
                    fit_time = t.elapsed
                except Exception as e:
                    acc = np.nan
                    fit_time = np.nan
                    print(f"    WARNING: {clf_name} failed: {e}")

                rows.append({
                    "dataset": dname,
                    "classifier": clf_name,
                    "run": run,
                    "accuracy": acc,
                    "classifier_train_time": fit_time,
                })

        # Print summary per dataset
        for clf_name in classifiers:
            accs = [
                r["accuracy"] for r in rows
                if r["dataset"] == dname and r["classifier"] == clf_name
                and not np.isnan(r["accuracy"])
            ]
            if accs:
                print(f"  {clf_name:>22}: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "improvement_classifiers.csv"))

    summary = df.groupby(["dataset", "classifier"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        train_time_mean=("classifier_train_time", "mean"),
    ).reset_index()
    save_results(
        summary, os.path.join(output_path, "improvement_classifiers_summary.csv"),
    )

    return df



# IMPROVEMENT 2: Z-NORMALISATION


def experiment_normalisation(dataset_names, data_path, output_path, num_runs=5):
    """
    Improvement 2: Test whether z-normalising time series before ROCKET helps

    Hypothesis: Normalising time series to zero mean and unit std before
    applying kernels should make ROCKET less sensitive to the scale of
    different datasets, potentially improving accuracy

    the paper's UCR code does NOT normalise, but the authors' own GitHub Repo
    documentation says to normalise. The scalability script DOES normalise, 
    this inconsistency suggests it might matter after all.

    For fair comparison, for each run we use the SAME kernels for both normalised
    and unnormalised data, so the only variable is the normalisation
    """
    print_header("IMPROVEMENT 2: Effect of Z-Normalisation")

    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)

        for run in range(num_runs):
            # Generate kernels ONCE (shared for fair comparison)
            kernels = generate_kernels(X_tr.shape[-1], 10_000)

            # Without normalisation (matching the paper's UCR code) 
            X_tr_t = apply_kernels(X_tr, kernels)
            X_te_t = apply_kernels(X_te, kernels)
            X_tr_t = np.nan_to_num(X_tr_t, nan=0.0, posinf=0.0, neginf=0.0)
            X_te_t = np.nan_to_num(X_te_t, nan=0.0, posinf=0.0, neginf=0.0)

            clf_no = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            )
            clf_no.fit(X_tr_t, Y_tr)
            acc_no_norm = clf_no.score(X_te_t, Y_te)

            # With z-normalisation 
            X_tr_norm = z_normalise(X_tr)
            X_te_norm = z_normalise(X_te)

            X_tr_t_norm = apply_kernels(X_tr_norm, kernels)
            X_te_t_norm = apply_kernels(X_te_norm, kernels)
            X_tr_t_norm = np.nan_to_num(X_tr_t_norm, nan=0.0, posinf=0.0, neginf=0.0)
            X_te_t_norm = np.nan_to_num(X_te_t_norm, nan=0.0, posinf=0.0, neginf=0.0)

            clf_yes = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            )
            clf_yes.fit(X_tr_t_norm, Y_tr)
            acc_with_norm = clf_yes.score(X_te_t_norm, Y_te)

            rows.append({
                "dataset": dname,
                "run": run,
                "accuracy_no_normalisation": acc_no_norm,
                "accuracy_with_normalisation": acc_with_norm,
                "difference": acc_with_norm - acc_no_norm,
            })

        # Summary for this dataset
        diffs = [r["difference"] for r in rows if r["dataset"] == dname]
        mean_diff = np.mean(diffs)
        direction = (
            "helps" if mean_diff > 0.001
            else ("hurts" if mean_diff < -0.001 else "no effect")
        )
        print(f"  Effect of normalisation: {mean_diff:+.4f} ({direction})")

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "improvement_normalisation.csv"))

    summary = df.groupby("dataset").agg(
        mean_no_norm=("accuracy_no_normalisation", "mean"),
        mean_with_norm=("accuracy_with_normalisation", "mean"),
        mean_difference=("difference", "mean"),
    ).reset_index()
    save_results(
        summary, os.path.join(output_path, "improvement_normalisation_summary.csv"),
    )

    return df



# IMPROVEMENT 3: SEED STABILITY (30 RUNS)


def experiment_stability(dataset_names, data_path, output_path, num_runs=30):
    """
    Improvement 3: Extended stability analysis with 30 independent runs

    Hypothesis: The paper reports standard deviations from 10 runs, suggesting
    ROCKET is more or less stable. We test this more rigorously with 30 runs to get a
    reliable estimate of the full distribution of accuracy values

    this matters because if a researcher runs ROCKET once due to computational or time limitations,
    how confident can they be that their single result is representative?
    """
    print_header("IMPROVEMENT 3: Seed Stability (30 Runs)")

    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)
        input_length = X_tr.shape[-1]

        accuracies = np.zeros(num_runs)

        for run in range(num_runs):
            kernels = generate_kernels(input_length, 10_000)
            X_tr_t = apply_kernels(X_tr, kernels)
            X_te_t = apply_kernels(X_te, kernels)

            X_tr_t = np.nan_to_num(X_tr_t, nan=0.0, posinf=0.0, neginf=0.0)
            X_te_t = np.nan_to_num(X_te_t, nan=0.0, posinf=0.0, neginf=0.0)

            clf = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            )
            clf.fit(X_tr_t, Y_tr)
            accuracies[run] = clf.score(X_te_t, Y_te)

            rows.append({
                "dataset": dname,
                "run": run,
                "accuracy": accuracies[run],
            })

        print(
            f"  Mean: {accuracies.mean():.4f}  Std: {accuracies.std():.4f}  "
            f"Min: {accuracies.min():.4f}  Max: {accuracies.max():.4f}  "
            f"Range: {accuracies.max() - accuracies.min():.4f}"
        )

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "improvement_stability.csv"))

    summary = df.groupby("dataset").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        accuracy_range=("accuracy", lambda x: x.max() - x.min()),
    ).reset_index()
    save_results(
        summary, os.path.join(output_path, "improvement_stability_summary.csv"),
    )

    return df



# IMPROVEMENT 4: FEATURE SELECTION (LOW-VARIANCE REMOVAL)


def experiment_feature_selection(dataset_names, data_path, output_path, num_runs=5):
    """
    Improvement 4: Remove low variance features before classification

    Hypothesis: ROCKET generates 20000 features, but many have near-zero
    variance (they give nearly the same value for every training example) and
    these features contribute noise rather than signal and removing them may:
    1. speed up classifier training and
    2. improve accuracy by reducing overfitting to noise

    We test several variance thresholds and compare them against the baseline
    (no feature selection, using all 20000 features)
    """
    print_header("IMPROVEMENT 4: Feature Selection (Variance Threshold)")

    # Thresholds to test:
    # None  = baseline (no feature removal)
    # 0.0   = remove only constant features (zero variance)
    # 0.001 = remove near constant features
    # 0.01  = remove low variance features more aggressively
    thresholds = [None, 0.0, 0.001, 0.01]

    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)

        for run in range(num_runs):
            # Generate and transform ONCE (shared across thresholds)
            kernels = generate_kernels(X_tr.shape[-1], 10_000)
            X_tr_t = apply_kernels(X_tr, kernels)
            X_te_t = apply_kernels(X_te, kernels)

            X_tr_t = np.nan_to_num(X_tr_t, nan=0.0, posinf=0.0, neginf=0.0)
            X_te_t = np.nan_to_num(X_te_t, nan=0.0, posinf=0.0, neginf=0.0)

            for threshold in thresholds:

                if threshold is not None:
                    # Apply variance threshold to remove low-variance features
                    selector = VarianceThreshold(threshold=threshold)
                    try:
                        X_tr_selected = selector.fit_transform(X_tr_t)
                        X_te_selected = selector.transform(X_te_t)
                        num_features_kept = X_tr_selected.shape[1]
                    except ValueError:
                        # If ALL features are removed (unlikely but possible)
                        X_tr_selected = X_tr_t
                        X_te_selected = X_te_t
                        num_features_kept = X_tr_t.shape[1]
                    label = f"threshold={threshold}"
                else:
                    # Baseline: no feature selection
                    X_tr_selected = X_tr_t
                    X_te_selected = X_te_t
                    num_features_kept = X_tr_t.shape[1]
                    label = "no_selection (baseline)"

                # Train and evaluate
                with Timer() as t:
                    clf = make_pipeline(
                        StandardScaler(),
                        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
                    )
                    clf.fit(X_tr_selected, Y_tr)
                acc = clf.score(X_te_selected, Y_te)

                rows.append({
                    "dataset": dname,
                    "run": run,
                    "threshold": label,
                    "accuracy": acc,
                    "num_features_kept": num_features_kept,
                    "num_features_total": X_tr_t.shape[1],
                    "fraction_kept": num_features_kept / X_tr_t.shape[1],
                    "classifier_train_time": t.elapsed,
                })

        # Print summary per dataset
        for threshold in thresholds:
            label = (
                f"threshold={threshold}"
                if threshold is not None
                else "no_selection (baseline)"
            )
            accs = [
                r["accuracy"] for r in rows
                if r["dataset"] == dname and r["threshold"] == label
            ]
            feats = [
                r["num_features_kept"] for r in rows
                if r["dataset"] == dname and r["threshold"] == label
            ]
            if accs:
                print(
                    f"  {label:>28}: acc={np.mean(accs):.4f}  "
                    f"features={int(np.mean(feats)):>5}/20000"
                )

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "improvement_feature_selection.csv"))

    summary = df.groupby(["dataset", "threshold"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        mean_features_kept=("num_features_kept", "mean"),
        mean_fraction_kept=("fraction_kept", "mean"),
        mean_train_time=("classifier_train_time", "mean"),
    ).reset_index()
    save_results(
        summary,
        os.path.join(output_path, "improvement_feature_selection_summary.csv"),
    )

    return df



# IMPROVEMENT 5: ENSEMBLE OF SMALLER ROCKET MODELS


def experiment_ensemble(dataset_names, data_path, output_path, num_runs=5):
    """
    Improvement 5: Ensemble of multiple smaller ROCKET models

    Hypothesis: Instead of having one ROCKET with 10000 kernels, running multiple
    independent smaller ROCKETs and combining predictions by majority voting
    might produce more stable and potentially more accurate results

    This might work because each smaller ROCKET captures a different random
    sample of patterns. Combining them via voting smooths out the randomness
    and leverages the diversity of different kernel sets.

    We compare:
    - Baseline: 1 x 10000 kernels (standard ROCKET)
    - Ensemble: 3 x 5000 kernels (majority vote)
    - Ensemble: 5 x 2000 kernels (same total features, more diversity)
    - Ensemble: 5 x 5000 kernels (more total compute, maximum diversity)
    """
    print_header("IMPROVEMENT 5: Ensemble of Smaller ROCKET Models")

    # Each config: (label, list of kernel counts per sub-model)
    configs = [
        ("baseline_1x10000", [10_000]),
        ("ensemble_3x5000", [5_000, 5_000, 5_000]),
        ("ensemble_5x2000", [2_000, 2_000, 2_000, 2_000, 2_000]),
        ("ensemble_5x5000", [5_000, 5_000, 5_000, 5_000, 5_000]),
    ]

    rows = []

    for dname in dataset_names:
        print_subheader(dname)
        X_tr, Y_tr, X_te, Y_te = load_ucr_dataset(dname, data_path)
        input_length = X_tr.shape[-1]

        for run in range(num_runs):

            for config_label, kernel_counts in configs:

                all_predictions = []

                with Timer() as t:
                    for k in kernel_counts:
                        # Each sub model gets its own random kernels
                        kernels = generate_kernels(input_length, k)
                        X_tr_t = apply_kernels(X_tr, kernels)
                        X_te_t = apply_kernels(X_te, kernels)

                        X_tr_t = np.nan_to_num(
                            X_tr_t, nan=0.0, posinf=0.0, neginf=0.0,
                        )
                        X_te_t = np.nan_to_num(
                            X_te_t, nan=0.0, posinf=0.0, neginf=0.0,
                        )

                        clf = make_pipeline(
                            StandardScaler(),
                            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
                        )
                        clf.fit(X_tr_t, Y_tr)

                        # Get predictions from this sub model
                        preds = clf.predict(X_te_t)
                        all_predictions.append(preds)

                total_time = t.elapsed

                # Combine predictions by majority voting
                if len(all_predictions) == 1:
                    # Baseline: single model, no voting needed
                    final_predictions = all_predictions[0]
                else:
                    # Majority vote for each test example
                    pred_matrix = np.array(all_predictions)
                    final_predictions = np.zeros(
                        pred_matrix.shape[1], dtype=np.int32,
                    )
                    for j in range(pred_matrix.shape[1]):
                        votes = pred_matrix[:, j]
                        vote_counts = Counter(votes)
                        final_predictions[j] = vote_counts.most_common(1)[0][0]

                accuracy = np.mean(final_predictions == Y_te)

                rows.append({
                    "dataset": dname,
                    "run": run,
                    "config": config_label,
                    "num_sub_models": len(kernel_counts),
                    "kernels_per_model": kernel_counts[0],
                    "total_kernels": sum(kernel_counts),
                    "accuracy": accuracy,
                    "total_time": total_time,
                })

        # Print summary per dataset
        for config_label, _ in configs:
            accs = [
                r["accuracy"] for r in rows
                if r["dataset"] == dname and r["config"] == config_label
            ]
            if accs:
                print(
                    f"  {config_label:>22}: "
                    f"{np.mean(accs):.4f} (+/- {np.std(accs):.4f})"
                )

    df = pd.DataFrame(rows)
    save_results(df, os.path.join(output_path, "improvement_ensemble.csv"))

    summary = df.groupby(["dataset", "config"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        mean_time=("total_time", "mean"),
    ).reset_index()
    save_results(
        summary, os.path.join(output_path, "improvement_ensemble_summary.csv"),
    )

    return df



# MAIN


def main():
    parser = argparse.ArgumentParser(
        description="ROCKET improvement experiments.",
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="results")
    parser.add_argument("--dataset_list", default=None)
    parser.add_argument(
        "--experiment",
        default="all",
        choices=[
            "all", "classifiers", "normalisation", "stability",
            "feature_selection", "ensemble",
        ],
    )
    parser.add_argument("--num_runs", type=int, default=5)

    args = parser.parse_args()

    if args.dataset_list:
        dataset_names = load_dataset_names(args.dataset_list)
    else:
        # Auto detect all datasets from the data directory
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

    if args.experiment in ("all", "classifiers"):
        experiment_classifiers(
            dataset_names, args.data_path, args.output_path, args.num_runs,
        )

    if args.experiment in ("all", "normalisation"):
        experiment_normalisation(
            dataset_names, args.data_path, args.output_path, args.num_runs,
        )

    if args.experiment in ("all", "stability"):
        stability_runs = 30 if args.experiment == "stability" else max(args.num_runs, 30)
        experiment_stability(
            dataset_names, args.data_path, args.output_path, stability_runs,
        )

    if args.experiment in ("all", "feature_selection"):
        experiment_feature_selection(
            dataset_names, args.data_path, args.output_path, args.num_runs,
        )

    if args.experiment in ("all", "ensemble"):
        experiment_ensemble(
            dataset_names, args.data_path, args.output_path, args.num_runs,
        )

    print_header("ALL IMPROVEMENT EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
