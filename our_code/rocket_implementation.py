"""
rocket_implementation.py - Our implementation of the ROCKET transform

This file contains our own implementation of the ROCKET algorithm as described
in Section 3 of the paper:
    Dempster, Petitjean, Webb (2020). "ROCKET: Exceptionally fast and accurate
    time series classification using random convolutional kernels."
    Data Mining and Knowledge Discovery, 34(5), 1454-1495.

The logic is true to the paper's description and produces equivalent results
to the authors' original rocket_functions.py. We wrote this independently to
showcase our understanding of the method.

ROCKET working (summary):
    1. Generate a large number (default 10000) of random convolutional kernels, where
       each kernel has random: length, weights, bias, dilation, and padding.
    2. Convolve each kernel with each input time series.
    3. From each convolution output, extract two features:
       - PPV (proportion of positive values): fraction of output values > 0
       - Max (global max pooling): the single largest output value
    4. This produces a feature vector of length (num_kernels x 2) per time series
    5. These features are then used to train a linear classifier (such as ridge)

Important notes:
    - We use Numba's @njit decorator for just-in-time compilation, which makes
      the convolution operations run at near C speed.
    - parallel=True in apply_kernels enables automatic parallelisation across
      time series examples
    - fastmath=True enables the compiler to use faster (but slightly less precise)
      floating-point operations

Authors: Shashank Sanjay Tomar & Manan Malik
"""

import numpy as np
from numba import njit, prange



# KERNEL GENERATION

@njit(
    "Tuple((float64[:], int32[:], float64[:], int32[:], int32[:]))(int64, int64)"
)
def generate_kernels(input_length, num_kernels):
    """
    Generate a set of random convolutional kernels for the ROCKET transform

    Each kernel has five random properties:
        - Length: randomly chosen from {7, 9, 11}
        - Weights: sampled from N(0,1), then mean centered
        - Bias: sampled from U(-1, 1)
        - Dilation: sampled so the kernel can span up to the full input length
        - Padding: randomly chosen as zero-padding or no padding (50/50)

    Parameters:
    
    input_length : int
        The number of time steps in each input time series
    num_kernels : int
        How many random kernels to generate (default in the paper: 10000)

    Returns:
    
    weights : np.ndarray (float64)
        All kernel weights concatenated into a single flat array
        The weights for kernel i start at index sum(lengths[:i])
    lengths : np.ndarray (int32)
        The length of each kernel (7, 9, or 11)
    biases : np.ndarray (float64)
        The bias value for each kernel
    dilations : np.ndarray (int32)
        The dilation value for each kernel
    paddings : np.ndarray (int32)
        The padding amount for each kernel (0 or half the effective length)
    """
    # The three kernel lengths
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)

    # Randomly assign a length to each kernel
    lengths = np.random.choice(candidate_lengths, num_kernels)

    # Pre allocate arrays
    # Weights are stored in a single flat array for efficiency
    # Total number of weight values = sum of all kernel lengths
    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    # Track position in the flat weights array
    weight_index = 0

    for i in range(num_kernels):

        current_length = lengths[i]

        # Weights
        # Sample from standard normal distribution
        kernel_weights = np.random.normal(0, 1, current_length)
        # Mean center the weights so they sum to roughly 0
        # This makes the kernel insensitive to the overall level of the signal
        kernel_weights = kernel_weights - kernel_weights.mean()

        # Store in the flat array
        weight_end = weight_index + current_length
        weights[weight_index:weight_end] = kernel_weights

        # Bias
        # Sample from uniform(-1, 1)
        # Bias shifts the convolution output, affecting which values are
        # positive (important for the PPV feature)
        biases[i] = np.random.uniform(-1, 1)

        # Dilation
        # Dilation controls the spacing between kernel weight positions.
        # The formula ensures the dilated kernel can span up to the full
        # input length: with dilation d, a kernel of length l spans
        # (l-1)*d + 1 time steps
        # Maximum useful dilation is (input_length - 1) / (length - 1)
        # We sample the exponent uniformly in log space so small and large
        # dilations are equally likely on a logarithmic scale
        max_dilation_log2 = np.log2((input_length - 1) / (current_length - 1))
        dilation = 2 ** np.random.uniform(0, max_dilation_log2)
        dilation = np.int32(dilation)
        dilations[i] = dilation

        # Padding
        # With  a 50% probability, apply zero padding so the kernel can detect
        # patterns at the very start/end of the time series
        # Padding amount = half the effective kernel span
        if np.random.randint(2) == 1:
            paddings[i] = ((current_length - 1) * dilation) // 2
        else:
            paddings[i] = 0

        # Move to next position in flat weights array
        weight_index = weight_end

    return weights, lengths, biases, dilations, paddings



# SINGLE KERNEL APPLICATION

@njit(fastmath=True)
def apply_single_kernel(time_series, weights, length, bias, dilation, padding):
    """
    Convolve a single kernel with a single time series and extract features

    This performs 1D convolution (with dilation and optional zero padding)
    and extracts two summary features from the output:
        - PPV: proportion of positive values (fraction of output > 0)
        - Max: global maximum of the convolution output

    Parameters:
    
    time_series : np.ndarray (float64), shape (input_length,)
        A single input time series
    weights : np.ndarray (float64), shape (length,)
        The kernel weights
    length : int
        The kernel length (number of weights)
    bias : float
        The kernel bias (added to each convolution output)
    dilation : int
        The spacing between weight positions during convolution
    padding : int
        Number of zero-padding positions added to each side

    Returns:
    
    ppv : float
        Proportion of positive values in the convolution output
        Ranges from 0.0 (no positive values) to 1.0 (all positive)
    max_val : float
        The maximum value in the convolution output
    """
    input_length = len(time_series)

    # Calculate the length of the convolution output
    # With padding on both sides and dilation, the output length
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    
    ppv_count = 0          # Count of positive values (for PPV)
    max_val = -np.inf      # Running maximum (starts at negative infinity)

    # The convolution loop
    # We slide the kernel across the (possibly padded) time series
    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        # Compute the dot product at this position (with bias)
        conv_sum = bias

        index = i
        for j in range(length):
            # Only include values that fall within the actual time series
            # (positions outside are treated as zero, that's the zero-padding)
            if index > -1 and index < input_length:
                conv_sum = conv_sum + weights[j] * time_series[index]

            # Move to next position (skip by dilation amount)
            index = index + dilation

        # Update the running maximum
        if conv_sum > max_val:
            max_val = conv_sum

        # Count positive values (for PPV)
        if conv_sum > 0:
            ppv_count += 1

    # PPV = fraction of output values that were positive
    ppv = ppv_count / output_length

    return ppv, max_val



# BATCH KERNEL APPLICATION (ALL KERNELS × ALL TIME SERIES)


@njit(
    "float64[:,:](float64[:,:], Tuple((float64[::1], int32[:], float64[:], int32[:], int32[:])))",
    parallel=True,
    fastmath=True,
)
def apply_kernels(X, kernels):
    """
    Apply all kernels to all time series and return the feature matrix

    For each time series and each kernel, extracts 2 features (PPV and max),
    producing an output matrix of shape (num_examples, num_kernels * 2)

    The outer loop (over examples) is parallelised using Numba's prange,
    which automatically distributes work across all CPU cores

    Parameters:
    
    X : np.ndarray (float64), shape (num_examples, input_length)
        The input time series matrix and each row is one time series
    kernels : tuple
        The output of generate_kernels():
        (weights, lengths, biases, dilations, paddings)

    Returns:
    
    features : np.ndarray (float64), shape (num_examples, num_kernels * 2)
        The transformed feature matrix. For kernel j, the features are at
        columns [2*j] (PPV) and [2*j + 1] (max)
    """
    # Unpack the kernel tuple
    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    # Output matrix: 2 features per kernel (PPV and max)
    features = np.zeros((num_examples, num_kernels * 2), dtype=np.float64)

    # Parallel loop over examples (each example is independent)
    for i in prange(num_examples):

        # Track position in the flat weights array
        weight_index = 0
        # Track position in the output feature array
        feature_index = 0

        for j in range(num_kernels):

            # Extract this kernel weights from the flat array
            weight_end = weight_index + lengths[j]

            # Apply the kernel to this time series
            ppv, max_val = apply_single_kernel(
                X[i],
                weights[weight_index:weight_end],
                lengths[j],
                biases[j],
                dilations[j],
                paddings[j],
            )

            # Store the two features
            features[i, feature_index] = ppv
            features[i, feature_index + 1] = max_val

            # Advance positions
            weight_index = weight_end
            feature_index += 2

    return features
