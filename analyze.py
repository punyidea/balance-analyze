# coding=utf-8
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
import datetime, time
import calendar

#columns of interest in ball data
ball_starting_names = ('YYYY-MO-DD HH-MI-SS_SSS',
                       'GYROSCOPE X (rad/s)',
                       'GYROSCOPE Y (rad/s)',
                       'GYROSCOPE Z (rad/s)',
                       'ACCELEROMETER X (m/s²)',
                       'ACCELEROMETER Y (m/s²)',
                       'ACCELEROMETER Z (m/s²)',
                       'ORIENTATION X (pitch °)',
                       'ORIENTATION Y (roll °)',
                       'ORIENTATION Z (azimuth °)')
band_starting_names = ('timestamp(unix)',
                        'rotation_rate_x(radians/s)',
                        'rotation_rate_y(radians/s)',
                        'rotation_rate_z(radians/s)',
                        'user_acc_x(G)',
                        'user_acc_y(G)',
                        'user_acc_z(G)',
                        'attitude_pitch(radians)',
                        'attitude_roll(radians)',
                        'attitude_yaw(radians)'
                        )

final_storage_names = ('time',
                       'gyro x',
                       'gyro y',
                       'gyro z',
                       'accel x', #in Gs
                       'accel y', #in Gs
                       'accel z', #in Gs
                       'orientation x',
                       'orientation y',
                       'orientation z'
                      )



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
                            ball_time_str ='YYYY-MO-DD HH-MI-SS_SSS', band_time_str = 'timestamp(unix)',
                            ):
    '''
    Synchronizes data for the balance exercises
    :param ball_data:
    :param band_data:
    :return:
    '''

    band_data_times = band_data[band_time_str].as_matrix()
    ball_data_times = ball_data[ball_time_str]
    ball_data_times = [datetime.datetime.strptime(time_str,'%Y-%m-%d %H:%M:%S:%f') for time_str in ball_data_times]
    ball_data_times = np.array([float(time_struct.strftime("%s.%f")) for time_struct in ball_data_times])
    ball_data[ball_time_str] = ball_data_times

    min_time = np.maximum(np.min(ball_data_times),np.min(band_data_times))
    max_time = np.minimum(np.max(ball_data_times),np.max(band_data_times))


    band_data[band_time_str] = band_data[band_time_str] - min_time
    ball_data[ball_time_str] = ball_data[ball_time_str] - min_time

    band_data_indices = np.logical_and(min_time<= band_data_times,band_data_times<=max_time)

    band_data = band_data.loc[band_data_indices]
    ball_data = ball_data.loc[np.logical_and(min_time <= ball_data_times,ball_data_times <= max_time)]

    band_data.rename(columns = dict(zip(band_starting_names,final_storage_names)),inplace=True)

    ball_data.rename(columns= dict(zip(ball_starting_names, final_storage_names)),inplace=True)
    ball_data['accel x'] = ball_data['accel x']/9.81
    ball_data['accel y'] = ball_data['accel y'] / 9.81
    ball_data['accel z'] = ball_data['accel z'] / 9.81

    return band_data,ball_data



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
    band_data, ball_data =  synchronize_and_cleanup(ball_data,band_data)






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