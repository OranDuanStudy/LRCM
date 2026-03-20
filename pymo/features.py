'''
A set of mocap feature extraction functions

Created: Nov 17 2017
'''
import numpy as np
import pandas as pd
import peakutils
import matplotlib.pyplot as plt

def get_foot_contact_idxs(signal, t=0.02, min_dist=120):
    up_idxs = peakutils.indexes(signal, thres=t/max(signal), min_dist=min_dist)
    down_idxs = peakutils.indexes(-signal, thres=t/min(signal), min_dist=min_dist)

    return [up_idxs, down_idxs]
"""
Find foot contact event indices (heel strike and toe off) in a given signal.

Args:
    signal: Input signal (e.g., vertical ground reaction force)
    t: Threshold for peak detection. Default is 0.02
    min_dist: Minimum distance between peaks. Default is 120

Returns:
    List containing two arrays: [heel strike indices, toe off indices]
"""

def create_foot_contact_signal(mocap_track, col_name, start=1, t=0.02, min_dist=120):
    signal = mocap_track.values[col_name].values
    idxs = get_foot_contact_idxs(signal, t, min_dist)

    step_signal = []

    c = start
    for f in range(len(signal)):
        if f in idxs[1]:
            c = 0
        elif f in idxs[0]:
            c = 1

        step_signal.append(c)

    return step_signal
"""
Create a binary foot contact signal based on heel strike and toe off.

Args:
    mocap_track: MocapData object containing motion capture data
    col_name: Column name in mocap_track.values representing the signal (e.g., vertical ground reaction force)
    start: Initial value of foot contact signal (0 or 1). Default is 1
    t: Threshold for peak detection. Default is 0.02
    min_dist: Minimum distance between peaks. Default is 120

Returns:
    Binary foot contact signal (0 indicates toe off, 1 indicates heel strike)
"""


def plot_foot_up_down(mocap_track, col_name, t=0.02, min_dist=120):

    signal = mocap_track.values[col_name].values
    idxs = get_foot_contact_idxs(signal, t, min_dist)

    plt.plot(mocap_track.values.index, signal)
    plt.plot(mocap_track.values.index[idxs[0]], signal[idxs[0]], 'ro')
    plt.plot(mocap_track.values.index[idxs[1]], signal[idxs[1]], 'go')
"""
Plot the input signal and detected heel strikes (red) and toe offs (green).

Args:
    mocap_track: MocapData object containing motion capture data
    col_name: Column name in mocap_track.values representing the signal (e.g., vertical ground reaction force)
    t: Threshold for peak detection. Default is 0.02
    min_dist: Minimum distance between peaks. Default is 120
"""