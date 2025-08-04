import numpy as np

# Function to calculate the classical min-entropy
def classical_min_entropy(probabilities):
    if len(probabilities) != 2:
        raise ValueError("Input array must have length=2")
    probabilities_vector = np.array(probabilities)  # Ensure probabilities are in a numpy array
    H = -np.log2(max(probabilities_vector))
    return H

# Function to calculate the classical max-entropy
def classical_max_entropy(probabilities, d):
    if len(probabilities) != d:
        raise ValueError(f'Input array must have length={d}')
    probabilities_array = np.array(probabilities)  # Ensure probabilities are in a numpy array
    if np.any(probabilities_array < 0) or np.any(probabilities_array > 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    if not np.isclose(sum(probabilities_array), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    s = 0
    for i in range(len(probabilities_array)):
        s += np.sqrt(probabilities_array[i])
    H_max = 2 * np.log2(s)
    return H_max

# Function to calculate the quantum conditional min-entropy
def quantum_min_entropy(probabilities, d):
    probabilities = np.array(probabilities)  # Ensure probabilities are in a numpy array
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    if not np.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    H_min_quantum = np.log2(d) - classical_max_entropy(probabilities, d)
    return H_min_quantum


def find_quantum_min_entropy(basis_probabilities, d):
    basis_probabilities_vector = np.array(basis_probabilities)
    H_min_quantum = -1
    for i in range(len(basis_probabilities_vector)):
        if i % d == 0:
            probabilities_reduced = np.array(basis_probabilities_vector[i:(i+d)])
            if np.any(probabilities_reduced < 0) or np.any(probabilities_reduced > 1):
                raise ValueError("Probabilities must be between 0 and 1.")
            if not np.isclose(sum(probabilities_reduced), 1.0):
                raise ValueError("Probabilities must sum to 1.")
            H = quantum_min_entropy(probabilities_reduced, d)
            if H > H_min_quantum:
                H_min_quantum = H
    return H_min_quantum

def stokes_params_and_quantum_tomography_min_entropy(coincidences):   # Function to determine the stokes params
    intensities = {
    'H': coincidences[0],  # Horizontal
    'V': coincidences[1],  # Vertical
    'D': coincidences[2],  # Diagonal
    'A': coincidences[3],  # Anti-diagonal
    'L': coincidences[4],  # Left-circular
    'R': coincidences[5],  # Right-circular
    }
    # Calculate Stokes parameters
    S_0 = (intensities['H'] + intensities['V'])/(intensities['H'] + intensities['V'])
    S_1 = (intensities['H'] - intensities['V'])/(intensities['H'] + intensities['V'])
    S_2 = (intensities['D'] - intensities['A'])/(intensities['D'] + intensities['A'])
    S_3 = (intensities['R'] - intensities['L'])/(intensities['R'] + intensities['L'])


    # Find the quantum conditional min-entropy with tomography
    qc_H_min_entropy = -np.log2((1 + np.sqrt(1 - S_1**2 - S_2**2))/2)

    return S_0, S_1, S_2, S_3, qc_H_min_entropy

def density_matrix_calculator(coincidences):
    S_0, S_1, S_2, S_3, qc_H_min_entropy = stokes_params_and_quantum_tomography_min_entropy(coincidences)

    I = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    density_matrix = np.array(0.5 * (I + S_1 * sigma_x + S_2 * sigma_y + S_3 * sigma_z))

    return density_matrix