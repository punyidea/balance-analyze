from __future__ import division,print_function
##############################################
# Balance Analyzer
# Written by: Leo Meister, Victor Prieto
##############################################

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import time
import calendar


# we want col_nums to be [0:15]
def load_band_data(filename, col_nums = None):

    return pd.read_csv(filename, usecols=col_nums)

def load_ball_data(filename,col_nums = None):

    data_frame = pd.read_csv(filename, usecols=col_nums, sep = ';', skiprows=1)

    return data_frame

def plot_col(data, col_names, title, time_name):

    time = np.array(data[time_name])
    processed_data = filter(np.array([data[col_name] for col_name in col_names]))

    plt.plot(time, processed_data.T)
    plt.legend(col_names)
    plt.title(title)

def filter(data, filt_len = 25):

    return lfilter(np.ones(filt_len)/filt_len, [1], data, axis=1)

def synchronize_and_cleanup(ball_data,band_data,
                            ball_time_str ='YYYY-MO-DD HH-MI-SS_SSS', band_time_str = 'timestamp(unix)' ):
    '''
    Synchronizes data for the balance exercises
    :param ball_data:
    :param band_data:
    :return:
    '''

    band_data_times = band_data[band_time_str]
    ball_data_times = ball_data[ball_time_str]
    ball_data_times = [time.strptime(time_str,'%Y-%m-%d %H:%M:%S:%f') for time_str in ball_data_times]
    ball_data_times = np.array([calendar.timegm(time_struct) for time_struct in ball_data_times])



    pass



def avg_angle(gyro, period=None, target_angle=None):

    pass


def avg_angular_velocity(gyro, period=None):

    pass

def avg_angular_accel(gyro, period=None):

    pass




def process_two_feet(data):

    pass

def process_one_foot(data):

    pass

def process_target_angle(data):



    pass


if __name__ == '__main__':

#debugging section



    ball_data = load_ball_data('Data/Subject C/Ball Data/42 both feet day 1 trial 1.csv')
    band_data = load_band_data('Data/Subject C/Band Data/42 both feet day 1 trial 1.csv')
    #
    synchronize_and_cleanup(ball_data,band_data)






## end debugging section
    cols = [0, 4, 5, 6]     # gyroscope
    #cols = [0, 10, 11, 12]  # accerlerometer

    plt.figure()

    data = band_data = load_band_data('Data/Subject C/Band Data/42 target angle day 1 trial 1.csv',cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Chest - Accelerometer', time_name = 'timestamp(unix)')

    plt.figure()

    data = load_band_data('trial 1 Leo lift foot.csv', cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Ankle - Accelerometer', time_name='timestamp(unix)')

    plt.figure()

    data = load_band_data('trial 2 Leo lift foot.csv', cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Thigh - Accelerometer', time_name='timestamp(unix)')
    plt.show()