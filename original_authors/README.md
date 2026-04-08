# Original Authors Code and Results

This folder contains the **unmodified** code and published results from the original ROCKET paper:

> Dempster, A., Petitjean, F., & Webb, G. I. (2020). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery*, 34(5), 1454–1495.

**Source:** [https://github.com/angus924/rocket](https://github.com/angus924/rocket)

## Files

| File | Description |
|------|-------------|
| `rocket_functions.py` | Core ROCKET implementation (generate_kernels, apply_kernels) |
| `reproduce_experiments_ucr.py` | Authors script for UCR archive experiments |
| `reproduce_experiments_scalability.py` | Authors script for scalability experiments |

## Results

| File | Description |
|------|-------------|
| `results/results_ucr.csv` | Accuracy on 128 UCR datasets (default train/test split, 10 runs) |
| `results/results_ucr_resamples.csv` | Accuracy on 128 UCR datasets (resampled splits) |
| `results/results_scalability_training_set_size.csv` | Scalability by training set size |
| `results/results_scalability_time_series_length.csv` | Scalability by time series length |

## Important

These files are included **for reference only**. Our reproduction code is in the `our_code/` folder.
We wrote our own implementation of the ROCKET algorithm and our own experiment scripts.
These originals are kept so that anyone can verify our implementation matches the paper's approach.
