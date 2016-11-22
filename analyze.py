##############################################
# Balance Analyzer
# Written by: Leo Meister, Victor Prieto
##############################################

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter


# we want col_nums to be [0:15]
def load_data(filename, col_nums = None):

    return pd.read_csv(filename, usecols=col_nums)

def plot_col(data, col_names, title, time_name):

    time = np.array(data[time_name])
    processed_data = filter(np.array([data[col_name] for col_name in col_names]))

    plt.plot(time, processed_data.T)
    plt.legend(col_names)
    plt.title(title)

def filter(data):

    return lfilter(np.ones(25)/25, [1], data, axis=1)


if __name__ == '__main__':

    cols = [0, 4, 5, 6]     # gyroscope
    #cols = [0, 10, 11, 12]  # accerlerometer

    plt.figure()

    data = load_data('trial 3 42 lift foot.csv', cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Chest - Accelerometer', time_name = 'timestamp(unix)')

    plt.figure()

    data = load_data('trial 1 Leo lift foot.csv', cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Ankle - Accelerometer', time_name='timestamp(unix)')

    plt.figure()

    data = load_data('trial 2 Leo lift foot.csv', cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Thigh - Accelerometer', time_name='timestamp(unix)')
    plt.show()