import numpy as np
from numba import njit, prange


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(signal_length, num_kernels):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    selected_lengths = np.random.choice(candidate_lengths, num_kernels)

    all_weights = np.zeros(selected_lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    weight_cursor = 0

    for i in range(num_kernels):
        length = selected_lengths[i]
        weights = np.random.normal(0, 1, length)

        all_weights[weight_cursor:weight_cursor + length] = weights - weights.mean()
        biases[i] = np.random.uniform(-1, 1)

        max_dilation = (signal_length - 1) / (length - 1)
        dilation = 2 ** np.random.uniform(0, np.log2(max_dilation))
        dilations[i] = np.int32(dilation)

        paddings[i] = ((length - 1) * dilations[i]) // 2 if np.random.randint(2) else 0

        weight_cursor += length

    return all_weights, selected_lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_single_kernel(signal, weights, length, bias, dilation, padding):
    signal_length = len(signal)
    result_length = (signal_length + (2 * padding)) - ((length - 1) * dilation)

    positive_count = 0
    max_activation = -np.inf

    for i in range(-padding, signal_length + padding - ((length - 1) * dilation)):
        conv_sum = bias
        index = i

        for j in range(length):
            if 0 <= index < signal_length:
                conv_sum += weights[j] * signal[index]
            index += dilation

        max_activation = max(max_activation, conv_sum)
        if conv_sum > 0:
            positive_count += 1

    return positive_count / result_length, max_activation


@njit(
    "float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))",
    parallel=True,
    fastmath=True,
)
def apply_kernels(batch_signals, kernel_bank):
    weights, lengths, biases, dilations, paddings = kernel_bank
    num_samples, _ = batch_signals.shape
    num_kernels = len(lengths)

    transformed_features = np.zeros((num_samples, num_kernels * 2), dtype=np.float64)

    for sample_idx in prange(num_samples):
        w_cursor = 0
        f_cursor = 0

        for k_idx in range(num_kernels):
            w_end = w_cursor + lengths[k_idx]
            f_end = f_cursor + 2

            transformed_features[sample_idx, f_cursor:f_end] = apply_single_kernel(
                batch_signals[sample_idx],
                weights[w_cursor:w_end],
                lengths[k_idx],
                biases[k_idx],
                dilations[k_idx],
                paddings[k_idx],
            )

            w_cursor = w_end
            f_cursor = f_end

    return transformed_features
