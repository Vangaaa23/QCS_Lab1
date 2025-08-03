import pandas as pd
import numpy as np
from scipy.linalg import ishermitian
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


# Function to create a toepliz matrix
def toepliz_matrix_generation(n,m):
    """
    Generates a Toeplitz matrix of size n x m.

    Parameters:
        n (int): Number of rows in the matrix.
        m (int): Number of columns in the matrix.
    """
    first_row = np.random.randint(0, 2, m)
    first_column = np.random.randint(0, 2, n)
    first_column[0] = first_row[0]  # Ensure consistency

    toeplitz_matrix = np.zeros((n, m), dtype=int) # corrected the shape

    for i in range(n):
        for j in range(m):
            if i <= j:
                toeplitz_matrix[i, j] = first_row[j - i]
            else:
                toeplitz_matrix[i, j] = first_column[i - j]
    return toeplitz_matrix

# Defining leftover hashing parameter
def leftover_hashing_param(raw_data, h_min, k):
    """
    Computes the leftover hashing parameter.

    Parameters:
        raw_data ()
    """
    return 2 ** (-(h_min * len(raw_data) - k) / 2)

# Function to extract randomness using Toeplitz matrix
def extract_randomness(raw_data, toeplitz_matrix):
    raw_data_vector = np.array(raw_data)
    # Ensure the matrix dimensions match the data length
    if len(raw_data_vector) != toeplitz_matrix.shape[1]:
        raise ValueError("The raw data length must match the number of columns in the Toeplitz matrix.")

    # Perform matrix multiplication and return the result modulo 2
    extracted = np.dot(toeplitz_matrix, raw_data_vector) % 2
    return extracted

def leftover_hashing_length(raw_data, min_entropy, epsilon):
    """
    Computes the length of extracted randomness using the Leftover Hashing Lemma.

    Parameters:
        min_entropy (float): Min-entropy of the source.
        epsilon (float): Desired security parameter.

    Returns:
        float: Length of extracted randomness.
    """
    return math.floor(min_entropy * len(raw_data) - 2 * np.log2(1 / epsilon) + 2)