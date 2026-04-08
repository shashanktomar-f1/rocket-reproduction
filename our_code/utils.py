"""
utils.py - Shared utility functions for the entire ROCKET reproduction project

This module provides reusable helper functions for:
    - Loading UCR archive datasets from files
    - Z-normalising time series
    - Computing evaluation metrics
    - Saving and loading results
    - Comparing our results against the paper's results
    - Timing code blocks
    - Formatted printing for experiment logs

These are used by all four experiment scripts:
    reproduce_ucr.py, reproduce_scalability.py,
    sensitivity_analysis.py, improvements.py

Authors: Shashank Sanjay Tomar & Manan Malik
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score



# DATA LOADING


def load_ucr_dataset(dataset_name, data_path):
    """
    Load a single UCR dataset from files

    The UCR archive format:
        - Each dataset lives in its own folder
        - Two files: {name}_TRAIN and {name}_TEST
        - Each row = one time series
        - First column = class label, remaining columns = time series values

    Parameters:
    
    dataset_name : str
        Name of the dataset (such as "Adiac", "ArrowHead" etc.)
    data_path : str
        Parent directory containing all dataset folders

    Returns:
    
    X_train : np.ndarray, shape (n_train, ts_length)
    Y_train : np.ndarray, shape (n_train,)
    X_test  : np.ndarray, shape (n_test, ts_length)
    Y_test  : np.ndarray, shape (n_test,)
    """

    # Supporting both .txt and .tsv file extensions (different UCR archive versions)
    train_file = None
    test_file = None

    for ext in [".txt", ".tsv"]:
        candidate_train = os.path.join(data_path, dataset_name, f"{dataset_name}_TRAIN{ext}")
        candidate_test = os.path.join(data_path, dataset_name, f"{dataset_name}_TEST{ext}")
        if os.path.exists(candidate_train) and os.path.exists(candidate_test):
            train_file = candidate_train
            test_file = candidate_test
            break

    if train_file is None:
        raise FileNotFoundError(
            f"Dataset files not found for '{dataset_name}' in {data_path}. "
            f"Looked for .txt and .tsv extensions."
        )


    training_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)

    # First column is the label, rest are the time series values
    Y_train = training_data[:, 0].astype(np.int32)
    X_train = training_data[:, 1:]

    Y_test = test_data[:, 0].astype(np.int32)
    X_test = test_data[:, 1:]

    return X_train, Y_train, X_test, Y_test


def get_dataset_info(X_train, Y_train, X_test, Y_test):
    """
    Return a summary dictionary describing the dataset

    Useful for logging and for the report tables
    """
    return {
        "num_train": X_train.shape[0],
        "num_test": X_test.shape[0],
        "ts_length": X_train.shape[1],
        "num_classes": len(np.unique(Y_train)),
        "classes": sorted(np.unique(Y_train).tolist()),
    }


def load_dataset_names(filepath):
    """
    Load dataset names from a file (one name per line)

    Parameters:
    
    filepath : str
        Path to text file listing dataset names

    Returns:
    
    names : list of str
    """
    with open(filepath, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names



# NORMALISATION


def z_normalise(X):
    """
    Z-normalise each time series to zero mean and unit standard deviation

    Each row (time series) is normalised independently

    Parameters:
    
    X : np.ndarray, shape (n_samples, ts_length)

    Returns:
    
    X_normalised : np.ndarray, same shape
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    # Avoid division by zero for constant time series
    std[std == 0] = 1.0
    return (X - mean) / std



# EVALUATION


def compute_accuracy(Y_true, Y_pred):
    """
    Compute classification accuracy (fraction of correct predictions)

    This is the primary metric used in the ROCKET paper and the UCR benchmark
    """
    return accuracy_score(Y_true, Y_pred)



# RESULTS I/O


def save_results(results_df, filepath):
    """Save a results DataFrame to CSV, creating directories if needed"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results_df.to_csv(filepath)
    print(f"Results saved to: {filepath}")


def load_paper_results(filepath):
    """Load the authors' published results CSV, indexed by dataset name"""
    return pd.read_csv(filepath, index_col="dataset")


def compare_results(our_results, paper_results, metric="accuracy_mean"):
    """
    Build a side by side comparison table of our results vs the paper's results

    Parameters:
    
    our_results : pd.DataFrame
        Indexed by dataset name, must contain 'metric' column
    paper_results : pd.DataFrame
        Indexed by dataset name, must contain 'metric' column
    metric : str
        Column name to compare (default: "accuracy_mean")

    Returns:
    
    comparison : pd.DataFrame
        Columns: paper_{metric}, our_{metric}, difference, abs_difference
    """
    common = our_results.index.intersection(paper_results.index)

    comparison = pd.DataFrame(index=common)
    comparison[f"paper_{metric}"] = paper_results.loc[common, metric]
    comparison[f"our_{metric}"] = our_results.loc[common, metric]
    comparison["difference"] = (
        comparison[f"our_{metric}"] - comparison[f"paper_{metric}"]
    )
    comparison["abs_difference"] = comparison["difference"].abs()

    return comparison



# TIMING


class Timer:
    """
    Context manager for timing code block
    """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start



# FORMATTED PRINTING


def print_header(text, width=80, char="="):
    """Print a main header line for experiment sections"""
    print(f"\n{char * width}")
    print(f" {text} ".center(width, char))
    print(f"{char * width}")


def print_subheader(text, width=80, char="-"):
    """Print a subheader line for individual datasets"""
    print(f"{f' {text} ':{char}^{width}}")


def print_comparison_summary(comparison_df, metric="accuracy_mean"):
    """
    Print summary statistics comparing our results to the paper
    """
    print(f"COMPARISON SUMMARY ({metric})")
    print(f"Datasets compared:            {len(comparison_df)}")
    print(f"Mean absolute difference:     {comparison_df['abs_difference'].mean():.6f}")
    print(f"Max absolute difference:      {comparison_df['abs_difference'].max():.6f}")
    print(f"Datasets where we are higher: {(comparison_df['difference'] > 0).sum()}")
    print(f"Datasets where we are lower:  {(comparison_df['difference'] < 0).sum()}")
    print(f"Datasets where we match:      {(comparison_df['difference'] == 0).sum()}")
    print(f"Mean paper {metric}:   {comparison_df[f'paper_{metric}'].mean():.6f}")
    print(f"Mean our {metric}:     {comparison_df[f'our_{metric}'].mean():.6f}")

