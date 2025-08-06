import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import toeplitz
from typing import Sequence, Union, Optional


def generate_toeplitz_matrix(
    n: int,
    m: int,
    *,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    # Set up RNG
    if isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng()

    # first column and first row
    c0 = rng.integers(0, 2, size=n, dtype=np.uint8)
    r0 = rng.integers(0, 2, size=m, dtype=np.uint8)
    # enforce consistency at (0,0)
    r0[0] = c0[0]

    # build Toeplitz
    toepl = toeplitz(c0, r0).astype(np.uint8)
    return toepl

def leftover_hashing_epsilon(h_min: float, data_length: int, output_length: int) -> float:
    exponent = -(h_min * data_length - output_length) / 2
    return 2 ** exponent

def extract_random_bits(
    raw_bits: Sequence[int],
    toeplitz_matrix: np.ndarray
) -> np.ndarray:
    raw = np.asarray(raw_bits, dtype=np.uint8)
    n, m = toeplitz_matrix.shape
    if raw.ndim != 1 or raw.size != m:
        raise ValueError(f"raw_bits must be length {m}, got {raw.size}")
    # matrix multiplication mod 2
    product = toeplitz_matrix.dot(raw)
    return np.mod(product, 2).astype(np.uint8)


def max_extracted_length(
    h_min: float,
    data_length: int,
    epsilon: float
) -> int:
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be in (0,1)")
    raw_entropy = h_min * data_length
    subtract = 2 * math.log2(1 / epsilon)
    l = math.floor(raw_entropy - subtract)
    if l < 0:
        raise ValueError("Parameters yield negative extractable length")
    return l

def output_length_given_sec_param(H_min_array, epsilon_array, raw_bits_list):
    if len(H_min_array) != len(raw_bits_list):
        raise ValueError("H_min_array and raw_bits_list must have the same length.")

    output_matrix = []
    for H_min, raw_bits in zip(H_min_array, raw_bits_list):
        data_length = len(raw_bits)
        row = [
            max_extracted_length(H_min, data_length, eps)
            for eps in epsilon_array
        ]
        output_matrix.append(row)

    return output_matrix

def plot_bit_rate(security_params, output_length_given_sec_param, period, dataset_labels=None, _figsize=(6,6)):
    """
    Plots the relationship between security parameters and bitrate.
    """
    if not dataset_labels:
        dataset_labels = [
            'Trusted', 
            'EUP', 
            'TOMO', 
            '4-POVM', 
            '6-POVM'
        ]

    # Extract series in order 0,1,2,3,4
    series = [output_length_given_sec_param[i] for i in range(len(dataset_labels))]

    if any(len(s) != len(security_params) for s in series):
        raise ValueError("Each output-length series must have the same length as security_params.")

    plt.figure(figsize=_figsize)
    for i, current_output_length in enumerate(series):
        # compute bit-rate elementwise
        bit_rate = [L / period for L in current_output_length]
        plt.plot(security_params, bit_rate, marker='o', label=dataset_labels[i])

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xscale('log')
    plt.xlabel("Security Parameter [$\log_2 \\Delta$]")
    plt.ylabel("Bitrate [bit/s]")
    plt.title("Bitrate vs. Security Parameter")
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_output_matrix(output_matrix, epsilon_array, dataset_labels=None,):
    if not dataset_labels:
        dataset_labels = [
            'Trusted', 
            'EUP', 
            'TOMO', 
            '4-POVM', 
            '6-POVM'
        ]

    # Header
    header = ["Dataset \\ ε"] + [f"{eps:.0e}" for eps in epsilon_array]
    col_widths = [max(len(str(val)) for val in col) for col in zip(*([header] + [
        [label] + [str(val) for val in row] for label, row in zip(dataset_labels, output_matrix)
    ]))]

    # Print header
    print(" | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)))
    print("-+-".join("-" * w for w in col_widths))

    # Print rows
    for label, row in zip(dataset_labels, output_matrix):
        print(" | ".join(f"{val:<{w}}" for val, w in zip([label] + row, col_widths)))

def plot_output_lengths_vs_security(output_lengths, epsilon_array, H_min_array, dataset_labels=None, _figsize=(6,6)):
    if not dataset_labels:
        dataset_labels = [
            'Trusted', 
            'EUP', 
            'TOMO', 
            '4-POVM', 
            '6-POVM'
        ]

    plt.figure(figsize=_figsize)
    for i, output_row in enumerate(output_lengths):
        plt.plot(epsilon_array, output_row, marker='o', label=dataset_labels[i])

    plt.xscale('log')
    plt.xlabel("Security parameter $\\varepsilon$")
    plt.ylabel("Extractable output length $\\ell(\\varepsilon)$")
    plt.title("Extractable length vs Security parameter")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
