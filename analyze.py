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
import pickle
import csv

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
    plt.xlabel('Time Elapsed (s)')

def filter(data, filt_len = 25):

    return lfilter(np.ones(filt_len)/filt_len, [1], data, axis=1)


def synchronize_and_cleanup(ball_data,band_data,
                            ball_time_str ='YYYY-MO-DD HH-MI-SS_SSS', band_time_str = 'timestamp(unix)',
                            truncate_time=True, time_bug = False):
    '''
    Synchronizes data for the balance exercises
    :param ball_data:
    :param band_data:
    :return:
    '''

    band_data_times = band_data[band_time_str].as_matrix()
    ball_data_times = ball_data[ball_time_str].tolist()
    ball_data_times = [datetime.datetime.strptime(time_str,'%Y-%m-%d %H:%M:%S:%f') for time_str in ball_data_times]
    ball_data_times = np.array([float(time_struct.strftime("%s.%f")) for time_struct in ball_data_times])
    ball_data.loc[:,ball_time_str] = ball_data_times

    #compensate for PowerSense bug in recording time
    if time_bug:
        band_data.loc[:,band_time_str] = band_data_times = band_data_times + 1266.1

    min_time = np.maximum(np.min(ball_data_times),np.min(band_data_times))
    max_time = np.minimum(np.max(ball_data_times),np.max(band_data_times))



    if truncate_time:
        band_data_indices = np.logical_and(min_time<= band_data_times,band_data_times<=max_time)

        band_data = band_data.loc[band_data_indices]
        ball_data = ball_data.loc[np.logical_and(min_time <= ball_data_times,ball_data_times <= max_time)]

    band_data.loc[:,band_time_str] -=  min_time
    ball_data.loc[:,ball_time_str] -=  min_time

    #adjust band data
    band_data.rename(columns = dict(zip(band_starting_names,final_storage_names)),inplace=True)
    band_data.loc[:, 'orientation x'] = np.mod(band_data.loc[:, 'orientation x'], 2 * np.pi) * 360 / (2*np.pi)
    band_data.loc[:, 'orientation y'] = np.mod(band_data.loc[:, 'orientation y'], 2 * np.pi) * 360 / (2*np.pi)
    band_data.loc[:, 'orientation z'] = np.mod(band_data.loc[:, 'orientation z'], 2 * np.pi) * 360 / (2*np.pi)

    #adjust ball data
    ball_data.rename(columns= dict(zip(ball_starting_names, final_storage_names)),inplace=True)
    ball_data.loc[:,'accel x'] /= 9.81
    ball_data.loc[:,'accel y'] /= 9.81
    ball_data.loc[:,'accel z'] /= 9.81

    band_data.loc[:, 'time'] -= min_time
    ball_data.loc[:, 'time'] -= min_time


    return band_data,ball_data



def plot_trial_data(ball_data, band_data, trial_name):
    gyro_names = ['gyro x', 'gyro y', 'gyro z']
    orientation_names = ['orientation x', 'orientation y','orientation z']

    plt.subplot(2,2,1)
    plot_col(ball_data, gyro_names, '{} Gyro (ball)'.format(trial_name),'time')
    plt.ylabel('Angular Velocity (rad/s)')

    plt.subplot(2,2,3)
    plot_col(band_data,gyro_names,'{} Gyro (participant)'.format(trial_name),'time')
    plt.ylabel('Angular Velocity (rad/s)')

    plt.subplot(2,2,2)
    plot_col(ball_data, orientation_names, '{} Orientation (ball)'.format(trial_name), 'time')
    plt.ylabel('Orientation (degrees)')

    plt.subplot(2, 2,4)
    plot_col(band_data, orientation_names, '{} Orientation (participant)'.format(trial_name), 'time')
    plt.ylabel('Orientation (degrees)')

def plot_participant_data(subject_name, truncate_time = True):
    for filename in os.listdir('Data/{}/Ball Data'.format(subject_name)):
        with open('Data/{}/Subject data.pkl'.format(subject_name),'rb') as pickle_file:
            patient_info = pickle.load(pickle_file)
        if 'Icon'not in filename and filename in os.listdir('Data/{}/Band Data'.format(subject_name)):
            bugged_time = False
            for bugged_file in patient_info['time_bugged_files']:
                if bugged_file in filename:
                    bugged_time=True
                    break
            band_data = load_band_data('Data/{}/Band Data/{}'.format(subject_name,filename))
            ball_data = load_ball_data('Data/{}/Ball Data/{}'.format(subject_name,filename))

            band_data,ball_data = synchronize_and_cleanup(ball_data, band_data,
                                                          truncate_time=truncate_time, time_bug=bugged_time)
            plt.figure()
            plot_trial_data(ball_data,band_data,os.path.splitext(filename)[0])
    plt.show()



def avg_angle(data, period=None, target_angle=None):

    '''
    Returns the average angular displacement and average angular distance of the given trial
    :param data: dataframe containing data from a trial
    :param period: option - the period over which to average
    :param target_angle: optional - target angle to compare the averages against
    :return: (disp, dist) - a tuple of the average angular displacement and distance computed
    '''

    orient = data[['orientation x', 'orientation y', 'orientation z']]

    if period != None:
        orient = orient[period[0]:period[1], :]

    std_dev = orient.abs().std(axis = 0)
    return ('angular mean', orient.abs().mean(axis=0), 'angular deviation', std_dev, 'max angle', orient.max(axis=0), 'min angle', orient.abs().min(axis=0))


def avg_angular_speed(data, period=None):

    gyro = data[['gyro x', 'gyro y', 'gyro z']]

    if period != None:
        gyro = gyro[period[0]:period[1], :]

    point_speed = np.linalg.norm(gyro, axis=1)
    ang_speed = np.mean(point_speed)
    return ('angular speed', ang_speed)


def avg_angular_accel(data, period=None):

    gyro = data[['gyro x', 'gyro y', 'gyro z']]

    if period != None:
        gyro = gyro[period[0]:period[1], :]

    accel = np.diff(gyro, axis = 0)
    point_accel = np.linalg.norm(accel, axis=1)
    ang_accel = np.mean(point_accel)

    return ('angular acceleration', ang_accel)


def process_static(data):

    return (avg_angle(data), avg_angular_speed(data))


def process_dynamic(data, static_segment=None, dynamic_segment=None):

    data_times = data[['time']].as_matrix()
    static_data_indices = np.zeros((len(data_times),1))
    dynamic_data_indecs = np.zeros((len(data_times),1))

    for seg in static_segment:
        static_data_indices = np.logical_or(static_data_indices, np.logical_and(seg[0] <= data_times, data_times <= seg[1]))

    for seg in dynamic_segment:
        dynamic_data_indecs = np.logical_or(dynamic_data_indecs, np.logical_and(seg[0] <= data_times, data_times <= seg[1]))

    static = data.loc[np.squeeze(static_data_indices),:]
    dynamic = data.loc[np.squeeze(dynamic_data_indecs),:]

    return(process_static(static), (avg_angular_speed(dynamic), avg_angular_accel(dynamic)))


def process_subject(letter):

    band_dir = 'Data/Subject ' + letter + '/Band Data/'
    ball_dir = 'Data/Subject ' + letter + '/Ball Data/'

    band_summary = open(band_dir + 'band summary.txt', 'w')
    ball_summary = open(ball_dir + 'ball summary.txt', 'w')

    band_files = os.listdir(band_dir)
    ball_files = os.listdir(ball_dir)

    try:
        with open('Data/Subject ' + letter + '/Subject data.pkl') as pickle_file:
            subject_data = pickle.load(pickle_file)
            static = subject_data['static_times']
    except:
        static = None

    for file in band_files:

        print(file)
        if(file in ball_files):

            band_data = load_band_data(band_dir + file)
            ball_data = load_ball_data(ball_dir + file)
            band_data, ball_data = synchronize_and_cleanup(ball_data, band_data)


            if('both' in file or 'one' in file or 'calibration' in file):
                band_res = process_static(band_data)
                ball_res = process_static(ball_data)

            elif('target' in file):

                trial = 1

                if('trial 2' in file):
                    trial = 2

                if(static != None):
                    cur_static = static['target angle trial ' + str(trial)]
                    dyn = get_dynamic_intervals(cur_static)

                    band_res = process_dynamic(band_data, cur_static, dyn)
                    ball_res = process_dynamic(ball_data, cur_static, dyn)

            else:
                band_res = 'File does not match\n'
                ball_res = 'File does not match\n'

            band_summary.write(file + '\n')
            for t in band_res:
                band_summary.write('\n'.join(str(s) for s in t))
            band_summary.write('\n\n')

            ball_summary.write(file + '\n')
            for t in ball_res:
                ball_summary.write('\n'.join(str(s) for s in t))
            ball_summary.write('\n\n')

    band_summary.close()
    ball_summary.close()


def get_dynamic_intervals(static_int):

    dyn_int = []
    dyn_int.append([0,static_int[0][0]])

    for i in range(len(static_int)-1):
        dyn_int.append([static_int[i][1], static_int[i+1][0]])

    return dyn_int


if __name__ == '__main__':

    #section for figuring out mysterious time offset
    #plot_participant_data('Subject D')


    '''
    #debugging section
    gyro_names = ['gyro x', 'gyro y', 'gyro z']
    orientation_names = ['orientation x', 'orientation y', 'orientation z']
    #section for figuring out mysterious time offset
    plot_participant_data('Subject D',truncate_time=False)



    #debugging section
    gyro_names = ['gyro x', 'gyro y', 'gyro z']
    orientation_names = ['orientation x', 'orientation y', 'orientation z']

    #plot_participant_data('Subject C')
    trial_name = 'One Foot Fall'
    ball_data = load_ball_data('Data/Subject A/Ball Data/Subject A one foot fall.csv')

    #col_names = ['gyro x', 'gyro y' ,'gyro z']
    ball_data_times = ball_data[ball_starting_names[0]].tolist()
    ball_data_times = [datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S:%f') for time_str in ball_data_times]
    ball_data_times = np.array([float(time_struct.strftime("%s.%f")) for time_struct in ball_data_times])
    ball_data.loc[:, ball_starting_names[0]] = ball_data_times
    ball_data.rename(columns=dict(zip(ball_starting_names, final_storage_names)), inplace=True)

    #ball_data = load_ball_data('Data/Subject C/Ball Data/42 both feet day 1 trial 1.csv')
    #band_data = load_band_data('Data/Subject C/Band Data/42 both feet day 1 trial 1.csv')
    ball_data.loc[:, 'time'] -= np.min(ball_data_times)


    plt.subplot(2, 1, 1)
    plot_col(ball_data, gyro_names, '{} Gyro (ball)'.format(trial_name), 'time')
    plt.ylabel('Angular Velocity (rad/s)')

    plt.subplot(2, 1, 2)
    plot_col(ball_data, orientation_names, '{} Orientation (ball)'.format(trial_name), 'time')
    plt.ylabel('Orientation (degrees)')

    plt.figure()
    plot_participant_data('Subject D')


    '''

    #process_two_feet(ball_data)

    process_subject('E')

'''
## end debugging section
    cols = [0, 4, 5, 6]     # gyroscope
    #cols = [0, 10, 11, 12]  # accerlerometer

    plt.figure()

    data = band_data = load_band_data('trial 3 42 lift foot.csv',cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Chest - Gyro', time_name = 'timestamp(unix)')

    plt.figure()

    data = load_band_data('trial 1 Leo lift foot.csv', cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Ankle - Gyro', time_name='timestamp(unix)')

    plt.figure()

    data = load_band_data('trial 2 Leo lift foot.csv', cols)
    plot_col(data, data.columns.base[1:], 'Target Angle Exercise: Thigh - Gyro', time_name='timestamp(unix)')
    plt.show()
'''