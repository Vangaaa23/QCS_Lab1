import pandas as pd
import numpy as np
from math import inf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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

def get_probabilities(coincidences):
    probabilities = []
    for i in range(len(coincidences)):
        if i%2 == 0:
            probabilities.append(float(coincidences[i] / (coincidences[i] + coincidences[i + 1]))) 
        else: 
            probabilities.append(float(coincidences[i] / (coincidences[i] + coincidences[i - 1])))
    return probabilities

def analyze_coincidence_distribution(filename, resolution, window, min_duration=60, n_bins=100):
    """
    Analyze time difference distribution for coincidences and fit a Gaussian.

    Parameters:
    -----------
    filename : str
        Path to the file with raw measurement data (semicolon-separated, 2 columns).
    resolution : float
        Time resolution per unit (e.g., 80.955e-12 seconds).
    window : float
        Coincidence time window (in seconds).
    min_duration : float
        Duration in seconds for which to keep the data.
    n_bins : int
        Number of bins for the histogram.

    Returns:
    --------
    dict with amplitude, mean, std and their uncertainties
    """
    
    # Load data
    measurements = np.loadtxt(filename, delimiter=";").astype(np.uint64)
    timetags = measurements[:, 0] * resolution
    channels = measurements[:, 1]

    # Truncate to min_duration
    valid = timetags <= timetags[0] + min_duration
    timetags = timetags[valid]
    channels = channels[valid]

    # Compute time differences between events in different channels
    delta_time = []
    for i in range(1, len(timetags)):
        if channels[i] != channels[i - 1]:
            direction = 1 if channels[i] == 2 else -1  # channel 2 as reference
            delta_time.append((timetags[i] - timetags[i - 1]) * direction)
    delta_time = np.array(delta_time)

    # Select only coincidences within the time window
    mask = (delta_time >= -window) & (delta_time <= window)
    coincidences = delta_time[mask]

    # Histogram and bin centers
    counts, bins = np.histogram(coincidences, bins=n_bins)
    bins = (bins[:-1] + bins[1:]) / 2

    # Gaussian model
    def gaussian(x, amplitude, mean, std):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    # Initial parameter estimates
    initial_mean = np.sum(bins * counts) / np.sum(counts)
    initial_std = np.sqrt(np.sum(counts * (bins - initial_mean) ** 2) / np.sum(counts))
    initial_params = [np.max(counts), initial_mean, initial_std]

    # Gaussian fit
    (amplitude, mean, std), cov = curve_fit(gaussian, bins, counts, p0=initial_params)
    errors = np.sqrt(np.diag(cov))

    # Plot histogram and Gaussian
    x = np.linspace(min(coincidences), max(coincidences), 500)
    y = gaussian(x, amplitude, mean, std)
    plt.figure()
    plt.hist(coincidences, bins=n_bins, edgecolor='black', label='Data', stacked=True)
    plt.plot(x, y, 'r-', label='Gaussian fit')
    plt.xlabel('Time difference [s]')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    plt.title("Coincidence Time Difference Distribution")
    plt.show()

    # Scatter plot of time differences with ±3σ lines
    plt.figure()
    plt.scatter(range(len(coincidences)), coincidences, marker='.')
    plt.axhline(mean, color='r', label='Mean')
    plt.axhline(mean + 3 * std, color='g', linestyle='--', label='±3σ')
    plt.axhline(mean - 3 * std, color='g', linestyle='--')
    plt.xlabel('Index of time difference')
    plt.ylabel('Time difference [s]')
    plt.legend()
    plt.grid(True)
    plt.title("Coincidence Time Differences")
    plt.show()

    # Print results
    print(f'Amplitude = {amplitude:.2f} ± {errors[0]:.2f}')
    print(f'Mean = {mean:.2e} ± {errors[1]:.2e}')
    print(f'Std = {std:.2e} ± {errors[2]:.2e}')
    percent_within_3sigma = len(coincidences[(coincidences >= mean - 3 * std) & (coincidences <= mean + 3 * std)]) / len(coincidences) * 100
    print(f'Percentage within ±3σ: {percent_within_3sigma:.2f}%')

    return {
        "amplitude": (amplitude, errors[0]),
        "mean": (mean, errors[1]),
        "std": (std, errors[2]),
        "percentage_within_3sigma": percent_within_3sigma
    }
