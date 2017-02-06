from __future__ import division

import numpy as np


def bin_spikes(spike_times, min_time=None, max_time=None):
    """Convert arrays of spike times into a single binary array

    Converts spike times into binary arrays with 1 ms time bins

    Args:
    spike_times (list, N_units)
        A list of arrays in which each array contains the spike times
        (in seconds) for a single unit/trial
    min_time (float, default=None)
        Specify what time range to include (set the array width).
        If None, will use the floor of the earliest and ceil of the latest
        spike in all rows (in seconds)
    max_time (float, default=None)

    Returns:
    t_arr (N_timesteps)
        1D array representing the time bins
    spikes (N_units x N_timesteps)
        2D array in which each row is a single unit/trial and each column
        represents a one millisecond timestep
    """
    # map to milliseconds
    spike_times = [row * 1e3 for row in spike_times]
    rows = len(spike_times)

    if min_time is None:
        min_time = int(min(np.floor(row[0]) for row in spike_times))
    else:
        min_time = int(1e3 * min_time)

    if max_time is None:
        max_time = int(max(np.ceil(row[-1]) for row in spike_times))
    else:
        max_time = int(1e3 * max_time)

    # generate output array with correct dimensions
    spikes = np.zeros((rows, int(max_time - min_time)))

    for i, row in enumerate(spike_times):
        for spike_time in row[(row >= min_time) & (row <= max_time)]:
            spikes[i, int(np.floor(spike_time)) - min_time] = 1

    t_arr = np.linspace(min_time * 1e-3, max_time * 1e-3, spikes.shape[1])

    return t_arr, spikes

def gaussian_convolver(std):
    """Create gaussian in array form extending 3 std on either side

    Args:
    std (float):
        std of gaussian distribution

    convolver (fn):
        return a function that takes an array and convolves with gaussian
    """
    extend = 3 * np.floor(std)
    t = np.arange(-extend, extend + 1)
    std = float(std)
    gaussian = (
        pow(np.sqrt(2 * np.pi * pow(std, 2)), -1) *
        np.exp(-pow(t, 2) / (2 * pow(std, 2)))
    )
    return lambda d: np.convolve(d, gaussian, mode="same")


def exponential_convolver(tau):
    """Create normalized exponential fn in array form

    Args:
    tau (float):
        mean of exponential distribution

    convolver (fn):
        return a function that takes an array and convolves with exponential
    """
    t = np.arange(4 * tau)
    result = np.exp(-t / float(tau))
    exp = result / np.sum(result)

    # need to cut off the end since full will extend the last exponential past the end

    return lambda d: np.convolve(d, exp, mode="full")[:d.size]


def conv(data, convolver, *convolver_args):
    """Convolve an exponential with all rows in the dataset

    Args:
    data (N_samples x N_dim):
        array of datapoint coordinates
    convolver (fn)
        Use either gaussianConvolver or expConvolver
        This function takes args and returns an array for convolution
    *convolver_args (*args)
        Arguments to pass to convolver, i.e. tau for expConvolver

    Returns:
    convolved_data (N_samples, N_dim):
        array of datapoint coordinates after convolving
    """
    return np.apply_along_axis(convolver(*convolver_args), 1, data)

