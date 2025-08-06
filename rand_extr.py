import pandas as pd
import numpy as np
from pathlib import Path
import math
from scipy.linalg import toeplitz
from typing import Sequence, Union, Optional

##########################################################################

### FUNCTIONS ###

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

def extract_random_bits(
    raw_bits: Sequence[int],
    toeplitz_matrix: np.ndarray
) -> np.ndarray:
    raw = np.asarray(raw_bits, dtype=np.uint8)
    _, m = toeplitz_matrix.shape
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


def dataframe_creation(filename, resolution, duration):
    df = pd.read_csv(filename, delimiter=';', comment='#', names=['Time','Channel'], header=None)
    df['Time'] -= df['Time'][0]
    df['Time'] = df['Time'] * resolution
    df = df[df['Time'] <= duration]
    return df

def find_coincidence(df, delta_mean, delta_std, coincidences_num, coincidence_window):

    # Create two different arrays for 'Time' and 'Channel'
    # time_tags = df['Time']
    # channels = df['Channel']

    channel_changes = df["Channel"] != df["Channel"].shift()
    delta_time = df.loc[channel_changes, "Time"].diff()
    delta_time = (delta_time * df["Channel"].apply(lambda x: 1 if x == 3 else -1)).dropna()

    # Calculate mean and std of
    delta_mean.append(np.mean(delta_time))
    delta_std.append(np.std(delta_time))

    coincidence_mask = (delta_time >= -coincidence_window) & (delta_time <= coincidence_window)

    coincidences_num.append(coincidence_mask.sum())

    # Extract filtered time tags based on the coincidence mask
    filtered_time_tags = df.loc[df.index[channel_changes], "Time"][1:][coincidence_mask]

    # Return results as a DataFrame
    return filtered_time_tags.to_frame(name="Coinciding Time Tags")

def find_coincidence_for_list(file_list, 
                              df_names, 
                              resolution,
                              coincidence_window,
                              duration = 60):

    df_list = {}

    for file, df_name in zip(file_list, df_names):
        print(f"Creating dataframe for {file}")
        df_list[df_name] = dataframe_creation(file, resolution, duration)

    # Define the output list
    coincidences_list = {}

    # Define the output vectors
    delta_mean = []
    delta_std = []
    coincidences_num = []

    # Iterate through the DataFrames (values in df_list)
    for df_name, df in df_list.items():
        print(f"Searching for coincidences on in dataframe {df_name}")
        timed_coincidences = find_coincidence(df, 
                                              delta_mean, 
                                              delta_std, 
                                              coincidences_num, 
                                              coincidence_window=coincidence_window)
        coincidences_list[df_name] = timed_coincidences

    return coincidences_list, coincidences_num, delta_mean, delta_std

##########################################################################

### CONSTANTS ###
RESOLUTION = 80.955e-12  # [ps]
DURATION = 40.0 #60.0  # Max window of duration (s)
COINCIDENCE_WINDOW = 5e-10  # Window of coincidences in s
RAW_DATA_LENGTH = 1000

### FOLDERS ###
BASE_DIR = Path.cwd()
DATA_PATH = BASE_DIR / 'data'

### FILE'S NAME ###
projectors = ['H', 'V', 'D', 'A', 'L', 'R']
generated_states = ['diagonal'] #, 'mixed', 'psi', 'right']

files = []
df_names = []
for generated_state in generated_states:
    for projector in projectors:
        filename = f'{generated_state}_measured_on_{projector}.txt'
        files.append(DATA_PATH / filename)
        df_names.append(f'{generated_state}_on_{projector}')

coincidences_list, coincidences_num, _, _ = find_coincidence_for_list(files,
                                                                      df_names,
                                                                      resolution=RESOLUTION,
                                                                      coincidence_window=COINCIDENCE_WINDOW
                                                                      )

prepared_state = 'diagonal'

# Encoding the outcomes
coincidences_list[f'{prepared_state}_on_H']['Measure'] = '0'
coincidences_list[f'{prepared_state}_on_V']['Measure'] = '1'
coincidences_list[f'{prepared_state}_on_D']['Measure'] = '0'
coincidences_list[f'{prepared_state}_on_A']['Measure'] = '1'
coincidences_list[f'{prepared_state}_on_R']['Measure'] = '0'
coincidences_list[f'{prepared_state}_on_L']['Measure'] = '1'

# Merging encoded data and sorting by Time Tags
merged_dataframe_1 = pd.concat([coincidences_list[f'{prepared_state}_on_H'], 
                                coincidences_list[f'{prepared_state}_on_V']
                                ])
merged_dataframe_1.sort_values(by='Coinciding Time Tags', 
                               inplace=True, 
                               ascending=True)

merged_dataframe_2 = pd.concat([coincidences_list[f'{prepared_state}_on_H'], 
                                     coincidences_list[f'{prepared_state}_on_V'],
                                     coincidences_list[f'{prepared_state}_on_D'], 
                                     coincidences_list[f'{prepared_state}_on_A']
                                     ])
merged_dataframe_2.sort_values(by='Coinciding Time Tags', 
                                    inplace=True, 
                                    ascending=True)

merged_dataframe_3 = pd.concat([coincidences_list[f'{prepared_state}_on_H'], 
                                     coincidences_list[f'{prepared_state}_on_V'],
                                     coincidences_list[f'{prepared_state}_on_D'], 
                                     coincidences_list[f'{prepared_state}_on_A'],
                                     coincidences_list[f'{prepared_state}_on_R'], 
                                     coincidences_list[f'{prepared_state}_on_L']
                                     ])
merged_dataframe_3.sort_values(by='Coinciding Time Tags', 
                                    inplace=True, 
                                    ascending=True)

raw_data_1 = np.array(merged_dataframe_1['Measure'])
raw_data_1 = np.fromiter(''.join(raw_data_1), dtype=int)

raw_data_2 = np.array(merged_dataframe_2['Measure'])
raw_data_2 = np.fromiter(''.join(raw_data_2), dtype=int)

raw_data_3 = np.array(merged_dataframe_3['Measure'])
raw_data_3 = np.fromiter(''.join(raw_data_3), dtype=int)

raw_data_short_1 = raw_data_1[:RAW_DATA_LENGTH]
raw_data_short_2 = raw_data_2[:RAW_DATA_LENGTH]
raw_data_short_3 = raw_data_3[:RAW_DATA_LENGTH]

# Min-entropies
min_entropies = {
    'classical_Hm': 0.978,
    'quantum_Hm': 0.892,
    'quantum_tomo_Hm': 0.894,
    'POVM_square_Hm': 1,
    'POVM_octhaedron_Hm': 1.58,
}

raw_data = [
    raw_data_short_1,
    raw_data_short_2,
    raw_data_short_3,
    raw_data_short_2,
    raw_data_short_3
]

epsilon_array = [
    1e-5,
    1e-10,
    1e-15,
    1e-20, 
    1e-25,
    1e-30,
    1e-35
    ]

output_length_vector = []
for H_min, data in zip(list(min_entropies.values()), raw_data):
    output_length_vector.append(max_extracted_length(H_min, len(data), 1e-6))

outputs = []
for i, (output_length, data) in enumerate(zip(output_length_vector, raw_data)):
    
    toeplitz_matrix = generate_toeplitz_matrix(output_length, len(data))
    extracted_bits = extract_random_bits(data, toeplitz_matrix)
    outputs.append(extracted_bits)

# Etichette leggibili per ciascun dataset
labels = [
    "Classical",
    "Quantum",
    "Quantum Tomography",
    "Square POVM",
    "Octahedron POVM"
]

# Stampa ordinata dei risultati
for label, output in zip(labels, outputs):
    print(f'{label:<30}: {output}')