import pandas as pd
import numpy as np
from scipy.linalg import ishermitian
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import inf, sqrt

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
        df_list[df_name] = dataframe_creation(file, resolution, duration)

    # Define the output list
    coincidences_list = {}

    # Define the output vectors
    delta_mean = []
    delta_std = []
    coincidences_num = []

    # Iterate through the DataFrames (values in df_list)
    for df_name, df in df_list.items():
        timed_coincidences = find_coincidence(df, 
                                              delta_mean, 
                                              delta_std, 
                                              coincidences_num, 
                                              coincidence_window=coincidence_window)
        coincidences_list[df_name] = timed_coincidences

    return coincidences_list, coincidences_num, delta_mean, delta_std


def get_data_from_file(files_vec, resolution):
    experiments = []
    min_duration = inf
    for file in files_vec:
        measurements = np.loadtxt(file, delimiter = ";").astype(np.uint64)
        timetags = measurements[:,0] * resolution
        duration = timetags[-1] - timetags[0]
        if min_duration == 0 or duration < min_duration:
            min_duration = duration
        experiments.append((file, measurements, duration))
    return experiments, min_duration

def gaussian(x, amplitude, mean, std):
    """Funzione gaussiana."""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

def LHL_print(H_min_array, raw_data):
    security_params = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24, 1e-28]
    output_length = np.arange(1,10001)
    output_length_given_security_param = []
    security_param_given_output_length = []
    for sec_param in security_params:
        for data, H_min in zip(raw_data, H_min_array):
            output_length_given_security_param.append(leftover_hashing_length(data, H_min, ))