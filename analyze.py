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

def plot_col(data, col_names, title, time_name, filter_data = True):

    time = np.array(data[time_name])
    processed_data = np.array([data[col_name] for col_name in col_names])
    if filter_data:
        processed_data = filter(processed_data)


    plt.plot(time, processed_data.T)
    plt.legend(col_names)
    plt.title(title)
    plt.xlabel('Time Elapsed (s)')

def filter(data, filt_len = 25):

    return lfilter(np.ones(filt_len), [filt_len], data, axis=0)


def synchronize_and_cleanup(ball_data,band_data,
                            ball_time_str ='YYYY-MO-DD HH-MI-SS_SSS', band_time_str = 'timestamp(unix)',
                            truncate_time=True, time_bug = False, bugged_time = None):
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
        band_data.loc[:,band_time_str] = band_data_times = band_data_times + bugged_time

    min_time = np.maximum(np.min(ball_data_times),np.min(band_data_times))
    max_time = np.minimum(np.max(ball_data_times),np.max(band_data_times))



    if truncate_time:
        band_data_indices = np.logical_and(min_time<= band_data_times,band_data_times<=max_time)

        band_data = band_data.loc[band_data_indices]
        ball_data = ball_data.loc[np.logical_and(min_time <= ball_data_times,ball_data_times <= max_time)]

    #adjust band data
    band_data.rename(columns = dict(zip(band_starting_names,final_storage_names)),inplace=True)
    band_data.loc[:, 'orientation x'] = np.mod(band_data.loc[:, 'orientation x']  + np.pi, 2 * np.pi) * 360 / (2*np.pi)  - 180
    band_data.loc[:, 'orientation y'] = np.mod(band_data.loc[:, 'orientation y']  + np.pi, 2 * np.pi) * 360 / (2*np.pi)  - 180
    band_data.loc[:, 'orientation z'] = np.mod(band_data.loc[:, 'orientation z']  + np.pi, 2 * np.pi) * 360 / (2*np.pi)  - 180

    #adjust ball data
    ball_data.rename(columns= dict(zip(ball_starting_names, final_storage_names)),inplace=True)
    ball_data.loc[:,'accel x'] /= 9.81
    ball_data.loc[:,'accel y'] /= 9.81
    ball_data.loc[:,'accel z'] /= 9.81

    band_data.loc[:, 'time'] -= min_time
    ball_data.loc[:, 'time'] -= min_time


    return band_data,ball_data



def plot_trial_data(ball_data, band_data, trial_name):
    gyro_names = np.array(['gyro x', 'gyro y', 'gyro z'])
    orientation_names = np.array(['orientation x', 'orientation y','orientation z'])

    plt.subplot(2,2,1)
    plot_col(ball_data, gyro_names[[0,1]], '{} Gyro (ball)'.format(trial_name),'time')
    plt.ylabel('Angular Velocity (rad/s)')

    plt.subplot(2,2,3)
    plot_col(band_data,gyro_names[[0,2]],'{} Gyro (participant)'.format(trial_name),'time')
    plt.ylabel('Angular Velocity (rad/s)')

    plt.subplot(2,2,2)
    plot_col(ball_data, orientation_names[[0,1]], '{} Orientation (ball)'.format(trial_name), 'time', filter_data=False)
    plt.ylabel('Orientation (degrees)')

    plt.subplot(2, 2,4)
    plot_col(band_data, orientation_names[[0,2]], '{} Orientation (participant)'.format(trial_name), 'time', filter_data=False)
    plt.ylabel('Orientation (degrees)')

def plot_participant_data(subject_name, truncate_time = True):
    for filename in os.listdir('Data/{}/Ball Data'.format(subject_name)):
        with open('Data/{}/Subject data.pkl'.format(subject_name),'rb') as pickle_file:
            patient_info = pickle.load(pickle_file)
        if 'Icon'not in filename and filename in os.listdir('Data/{}/Band Data'.format(subject_name)):
            time_bug = False
            bugged_time = None
            for bugged_file in patient_info['time_bugged_files']:
                if bugged_file in filename:
                    time_bug = True
                    bugged_time=patient_info['time_bugged_files'][bugged_file]
                    break
            band_data = load_band_data('Data/{}/Band Data/{}'.format(subject_name,filename))
            ball_data = load_ball_data('Data/{}/Ball Data/{}'.format(subject_name,filename))

            band_data,ball_data = synchronize_and_cleanup(ball_data, band_data, truncate_time=truncate_time,
                                                          time_bug=bugged_time, bugged_time = bugged_time)
            plt.figure()
            plot_trial_data(ball_data,band_data,os.path.splitext(filename)[0])
    #plt.show()


'''
def avg_angle(data, source, period=None, target_angle=None):


    #Returns the average angular displacement and average angular distance of the given trial
    #:param data: dataframe containing data from a trial
    #:param period: option - the period over which to average
    #:param target_angle: optional - target angle to compare the averages against
    #:return: (disp, dist) - a tuple of the average angular displacement and distance computed


    orient = data[['orientation x', 'orientation y', 'orientation z']]
    #orient = filter(orient)

    if period != None:
        orient = orient[period[0]:period[1], :]

    orient = np.abs(np.array(orient))
    angle_mean = orient.mean(axis=0)
    std_dev = orient.std(axis = 0)

    temp = orient.max(axis=0)

    metrics = {source + ' orientation mean x': angle_mean[0], source + ' orientation mean y': angle_mean[1], source + ' orientation mean z': angle_mean[2], source + ' orientation deviation x': std_dev[0], source + ' orientation deviation y': std_dev[1], source + ' orientation deviation z': std_dev[2], source + ' max angle x': orient.max(axis=0)[0], source + ' max angle y': orient.max(axis=0)[1], source + ' max angle z': orient.max(axis=0)[2], source + ' min angle x': orient.min(axis=0)[0], source + ' min angle y': orient.min(axis=0)[1], source + ' min angle z': orient.min(axis=0)[2]}

    return metrics


def avg_angular_speed(data, source, period=None):

    gyro = data[['gyro x', 'gyro y', 'gyro z']]
    gyro = filter(gyro)

    if period != None:
        gyro = gyro[period[0]:period[1], :]

    point_speed = np.linalg.norm(gyro, axis=1) * 180 / np.pi
    ang_speed = np.mean(point_speed)
    speed_std = np.std(point_speed)
    return {source + ' mean angular speed': ang_speed, source + ' angular speed standard deviation': speed_std}


def avg_angular_accel(data, source, period=None):

    gyro = data[['gyro x', 'gyro y', 'gyro z']]
    gyro = filter(gyro)

    if period != None:
        gyro = gyro[period[0]:period[1], :]

    accel = np.diff(gyro, axis = 0) * 180 / np.pi
    point_accel = np.linalg.norm(accel, axis=1)
    ang_accel = np.mean(point_accel)
    accel_std = np.std(point_accel)
    metrics = {source + ' mean angular acceleration': ang_accel, source + ' angular acceleration standard deviation': accel_std}
    return metrics


def process_static(data, source):

    metrics = avg_angle(data, source)
    metrics.update(avg_angular_speed(data, source))

    return metrics


def process_dynamic(data, source, static_segment=None, dynamic_segment=None):

    data_times = data[['time']].as_matrix()
    static_data_indices = np.zeros((len(data_times),1))
    dynamic_data_indecs = np.zeros((len(data_times),1))

    for seg in static_segment:
        static_data_indices = np.logical_or(static_data_indices, np.logical_and(seg[0] <= data_times, data_times <= seg[1]))

    for seg in dynamic_segment:
        dynamic_data_indecs = np.logical_or(dynamic_data_indecs, np.logical_and(seg[0] <= data_times, data_times <= seg[1]))

    static = data.loc[np.squeeze(static_data_indices),:]
    dynamic = data.loc[np.squeeze(dynamic_data_indecs),:]

    ang_speed = avg_angular_speed(dynamic, source)
    ang_speed.update(avg_angular_accel(dynamic, source))

    return(process_static(static, source), ang_speed)

'''
def get_segments(data, static_segment, dynamic_segment):

    data_times = data[['time']].as_matrix()
    static_data_indices = np.zeros((len(data_times), 1))
    dynamic_data_indecs = np.zeros((len(data_times), 1))

    for seg in static_segment:
        static_data_indices = np.logical_or(static_data_indices, np.logical_and(seg[0] <= data_times, data_times <= seg[1]))

    for seg in dynamic_segment:
        dynamic_data_indecs = np.logical_or(dynamic_data_indecs, np.logical_and(seg[0] <= data_times, data_times <= seg[1]))

    static = data.loc[np.squeeze(static_data_indices),:]
    dynamic = data.loc[np.squeeze(dynamic_data_indecs),:]

    return (static, dynamic)


# Only use Y axis for target angle ball, only use Z axis for target angle band for all measurements
# Compute skewness for precision (ratio of the means)
# avg angular deviation, max absolute deviation
# average over all exercises for each metric

# Stability - avg angular speed
# Precision - avg angualar deviation, max deviation, skewness is y/x or x/y (whichever is greater than 0, multiply by -1 is x > y)
# Acceleration - avg angular acceleration

def compute_stability(data, cols):

    data = filter(data[cols])
    point_speed = np.linalg.norm(data, axis=1) * 180 / np.pi
    avg_speed = np.mean(point_speed)

    return avg_speed

def compute_fluidity(data, cols):

    data = filter(data[cols])
    accel = np.diff(data, axis=0) * 180 / np.pi
    point_accel = np.linalg.norm(accel, axis=1)
    avg_accel = np.mean(point_accel)

    return avg_accel


def compute_precision(data, cols, target_angle = None):
    '''
    negative skewness indicates mean of absolute deviations x > y
    positive skewness indicates mean y > x

    :param data:
    :param target_angle:
    :return:
    '''

    data = np.abs(data[cols])

    if(target_angle != None):
        combined = np.array(data[['orientation y']])
        deviation = np.sqrt(np.sum(np.power(combined - target_angle, 2)) / data.size)
        y_dev = deviation

    else:
        combined = np.linalg.norm(data, axis=1)
        deviation = np.std(combined, axis=0)
        target_angle = np.mean(combined, axis=0)
        y_dev = np.std(np.array(data[['orientation y']]))

    x_dev = np.std(np.array(data[['orientation x']]))
    skewness = y_dev / x_dev

    if(np.abs(skewness) < 1):
        skewness = -1/skewness

    abs_dev = np.abs(combined - target_angle)
    max_abs_dev = np.max(abs_dev)

    return deviation, max_abs_dev, skewness


def process_subject(letter, restricted = False, restriction = None):

    band_dir = 'Data/Subject ' + letter + '/Band Data/'
    ball_dir = 'Data/Subject ' + letter + '/Ball Data/'

    band_summary = open(band_dir + 'band summary.txt', 'w')
    ball_summary = open(ball_dir + 'ball summary.txt', 'w')

    band_files = os.listdir(band_dir)
    ball_files = os.listdir(ball_dir)

    frame_static = pd.DataFrame()
    frame_dyn = pd.DataFrame()
    metrics_frame = pd.DataFrame()

    precision_cols_ball = ['orientation x', 'orientation y' ]
    stability_cols_ball = ['gyro x', 'gyro y']
    fluidity_cols_ball = ['gyro x', 'gyro y']
    stability_cols_band = ['gyro z']
    fluidity_cols_band = ['gyro z']

    num_dynamic = 0
    num_static = 0

    metrics = {'deviation': 0, 'max deviation': 0, 'skewness': 0, 'absolute skewness': 0, 'stability': 0, 'fluidity': 0}

    with open('Data/Subject ' + letter + '/Subject data.pkl') as pickle_file:
            subject_data = pickle.load(pickle_file)
            try:
                static = subject_data['static_times']
            except:
                static = None
            try:
                bugged_files = subject_data['time_bugged_files']
            except:
                bugged_files = None

    for file in band_files:

        time_bug = False
        bugged_time = None

        for key in bugged_files:

            if(key in file):
                time_bug = True
                bugged_time = bugged_files[key]
                break

        file_dict_static = {'file name': file}
        file_dict_dynamic = {'file name': file}

        print(file)

        if(file in ball_files) and 'Icon' not in file and (not restricted or (restricted and restriction in file)):

            band_data = load_band_data(band_dir + file)
            ball_data = load_ball_data(ball_dir + file)
            band_data, ball_data = synchronize_and_cleanup(ball_data, band_data,
                                                           time_bug=time_bug,bugged_time=bugged_time)

            if('both' in file or 'one' in file or 'calibration' in file):

                precision = compute_precision(ball_data, precision_cols_ball)

                metrics['stability'] += compute_stability(ball_data, stability_cols_ball)
                metrics['stability'] += compute_stability(band_data, stability_cols_band)

                num_static += 2

                #band_res = process_static(band_data, 'band')
                #ball_res = process_static(ball_data, 'ball')
                #file_dict_static.update(band_res)
                #file_dict_static.update(ball_res)

            elif('target' in file):

                trial = 1

                if('trial 2' in file):
                    trial = 2

                if('Subject C' in file or 'Subject D' in file):
                    day_num = file[27]
                    trial_str = 'target angle day ' + day_num +  ' trial '
                else:
                    trial_str = 'target angle trial '

                if(static != None):

                    cur_static = static[trial_str + str(trial)]
                    dyn = get_dynamic_intervals(cur_static)

                    ball = get_segments(ball_data, cur_static, dyn)
                    band = get_segments(band_data, cur_static, dyn)

                    precision = compute_precision(ball[0], precision_cols_ball, target_angle=25)

                    metrics['stability'] += compute_stability(ball[0], stability_cols_ball)
                    metrics['stability'] += compute_stability(band[0], stability_cols_band)

                    metrics['fluidity'] += compute_fluidity(band[1], fluidity_cols_band)
                    metrics['fluidity'] += compute_fluidity(ball[1], fluidity_cols_ball)

                    num_dynamic += 2
                    num_static += 2
                    '''
                    band_res = process_dynamic(band_data, 'band', cur_static, dyn)
                    ball_res = process_dynamic(ball_data, 'ball', cur_static, dyn)
                    file_dict_static.update(band_res[0])
                    file_dict_dynamic.update(band_res[1])
                    file_dict_static.update(ball_res[0])
                    file_dict_dynamic.update(ball_res[1])
                    '''
            else:
                band_res = 'File does not match\n'
                ball_res = 'File does not match\n'
                print('something might be wrong')
                precision = (0,0,0)

            metrics['deviation'] += precision[0]
            metrics['max deviation'] += precision[1]
            metrics['skewness'] += precision[2]
            metrics['absolute skewness'] += np.abs(precision[2])

            '''
            band_summary.write(file + '\n')
            for t in band_res:
                band_summary.write('\n'.join(str(s) for s in t))
            band_summary.write('\n\n')

            ball_summary.write(file + '\n')
            for t in ball_res:
                ball_summary.write('\n'.join(str(s) for s in t))
            ball_summary.write('\n\n')
            '''


            #frame_static = frame_static.append(file_dict_static, ignore_index=True)
            #frame_dyn = frame_dyn.append(file_dict_dynamic, ignore_index=True)

    band_summary.close()
    ball_summary.close()

    metrics['deviation'] /= num_static / 2
    metrics['max deviation'] /= num_static / 2
    metrics['skewness'] /= num_static / 2
    metrics['absolute skewness'] /= np.sign(metrics['skewness']) * num_static / 2
    metrics['stability'] /= num_static

    try:
        metrics['fluidity'] /= num_dynamic
    except ZeroDivisionError:
        metrics['fluidity'] = None

    metrics_frame = metrics_frame.append(metrics, ignore_index=True)
    metrics_frame.to_csv('Data/Subject ' + letter + '/' + letter + ' metric summary.csv')

    #frame_static.to_csv('Data/Subject ' + letter + '/' + letter + ' static summary.csv')
    #frame_dyn.to_csv('Data/Subject ' + letter + '/' + letter + ' dynamic summary.csv')


def get_dynamic_intervals(static_int):

    dyn_int = []
    dyn_int.append([0,static_int[0][0]])

    for i in range(len(static_int)-1):
        dyn_int.append([static_int[i][1], static_int[i+1][0]])

    return dyn_int


if __name__ == '__main__':

    #section for figuring out mysterious time offset
    #plot_participant_data('Subject A')

    restriction = 'target angle'
    day = ' day 2'

    #process_subject('A', restricted=True, restriction=restriction)
    #process_subject('B', restricted=True, restriction=restriction)
    process_subject('C', restricted=True, restriction=restriction)# + day)
    process_subject('D', restricted=True, restriction=restriction)# + day)
    #process_subject('E', restricted=True, restriction=restriction)

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