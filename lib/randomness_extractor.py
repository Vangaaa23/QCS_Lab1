import numpy as np
import warnings
import math
from scipy.linalg import toeplitz
from typing import Sequence, Union, Optional


def generate_toeplitz_matrix(
    n: int,
    m: int,
    *,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """
    Generate an n×m binary Toeplitz matrix (entries in {0,1}).

    Parameters:
    -----------
    n : int
        Number of rows.
    m : int
        Number of columns.
    random_state : int | np.random.Generator, optional
        Seed or RNG for reproducibility. If int, uses np.random.default_rng(random_state).

    Returns:
    --------
    toepl : np.ndarray
        An array of shape (n, m) with dtype=np.uint8 and entries 0 or 1.
    """
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
    """
    Compute the security parameter ε for a given min-entropy extractor.

    From Leftover Hash Lemma:
        ε = 2^{-(H_min * n - k)/2}

    Parameters:
    -----------
    h_min : float
        Min-entropy rate (bits of entropy per input bit).
    data_length : int
        Length n of the raw input (number of bits).
    output_length : int
        Number k of output bits.

    Returns:
    --------
    epsilon : float
        The resulting statistical distance.
    """
    exponent = -(h_min * data_length - output_length) / 2
    return 2 ** exponent


def extract_random_bits(
    raw_bits: Sequence[int],
    toeplitz_matrix: np.ndarray
) -> np.ndarray:
    """
    Perform bit-wise randomness extraction via a binary matrix multiply mod 2.

    Parameters:
    -----------
    raw_bits : sequence of {0,1}
        The input bit-string of length m = toeplitz_matrix.shape[1].
    toeplitz_matrix : np.ndarray
        A binary matrix of shape (n, m).

    Returns:
    --------
    extracted_bits : np.ndarray
        An array of length n with entries 0 or 1.
    """
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
    """
    Compute the maximum number of extractable bits ℓ according to the Leftover Hash Lemma:

        ℓ ≤ H_min * n - 2·log2(1/ε)

    We take the floor to get an integer.

    Parameters:
    -----------
    h_min : float
        Min-entropy rate (bits per input bit).
    data_length : int
        Number n of input bits.
    epsilon : float
        Desired security parameter (statistical distance).

    Returns:
    --------
    ℓ : int
        Maximum extractable length.
    """
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be in (0,1)")
    raw_entropy = h_min * data_length
    subtract = 2 * math.log2(1 / epsilon)
    ℓ = math.floor(raw_entropy - subtract)
    if ℓ < 0:
        raise ValueError("Parameters yield negative extractable length")
    return ℓ

def safe_max_length(h_min: float, n: int, epsilon: float) -> int:
    """
    Like max_extracted_length, but returns 0 if parameters yield negative length.
    """
    try:
        return max_extracted_length(h_min, n, epsilon)
    except ValueError:
        warnings.warn(
            f"Leftover-Hashing parameters yield ℓ<0 (H_min={h_min}, n={n}, ε={epsilon}). "
            "Setting ℓ = 0."
        )
        return 0


def safe_extract(raw_data, H_min, epsilon, *, random_state=None):
    """
    Compute extractable length ℓ = max_extracted_length(H_min, n, ε).
    If ℓ < 0, warn and return empty array.
    Otherwise, generate an ℓ×n Toeplitz matrix and extract bits.
    """
    n = len(raw_data)
    try:
        ℓ = max_extracted_length(H_min, n, epsilon)
    except ValueError as e:
        warnings.warn(
            f"Leftover-Hashing parameters yield ℓ<0 (H_min={H_min}, n={n}, ε={epsilon}). "
            "No bits extracted."
        )
        return np.zeros(0, dtype=np.uint8)

    if ℓ == 0:
        # nothing to extract
        return np.zeros(0, dtype=np.uint8)

    # now build the Toeplitz matrix and extract
    T = generate_toeplitz_matrix(ℓ, n, random_state=random_state)
    return extract_random_bits(raw_data, T)
