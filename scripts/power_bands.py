#!/usr/bin/env python

import argparse
import sys
import time
from glob import glob
import os
from os.path import join, basename, isfile, realpath, dirname

import h5py
import numpy as np
import tables
import scipy.io
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.data.downsample import downsample
from base.config_loader import ConfigLoader
from base.data.data_table import create_table_description, COLUMN_LABEL, COLUMN_MOUSE_ID, COLUMN_LAB


def parse():
    parser = argparse.ArgumentParser(description='data transformation script')
    parser.add_argument('--experiment', '-e', required=False, default='standard_config',
                        help='name of experiment to transform data to')

    return parser.parse_args()


def read_recording(mat_path: str, sr_old: int, labs_channels: dict, lab_name: str):
    rec = scipy.io.loadmat(mat_path)['signal']

    features = {}

    for channel in labs_channels[lab_name].keys():
        signal = rec[labs_channels[lab_name][channel]]

        s = downsample(signal,
                       sr_old=sr_old,
                       sr_new=config.SAMPLING_RATE,
                       fmax=0.4 * config.SAMPLING_RATE,
                       outtype='sos',
                       method='pad')
        
        s = (s - np.mean(s, keepdims=True)) / np.std(s, keepdims=True)

        features[channel] = s

    sample_start_times = np.arange(0, features[channel].shape[0], config.SAMPLING_RATE*config.SAMPLE_DURATION, dtype=int)

    return features, sample_start_times.tolist()


def read_labels(labels_path: str):
    labels_correspondence = {'1': 'Wake',
                            '2': 'Non REM',
                            '3': 'REM',
                            '4': 'Art'}
        
    if labels_path.split(os.sep)[-1][-3:] == 'mat':
        labels = scipy.io.loadmat(labels_path)[os.path.basename(labels_path).split('.')[0]]
    elif labels_path.split(os.sep)[-1][-3:] == 'csv':
        labels = pd.read_csv(labels_path, index_col=0).iloc[:,0].to_numpy()

    labels_str = [labels_correspondence[str(int(l))] for l in labels.squeeze()]

    return labels_str


def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """
    From https://raphaelvallat.com/bandpower.html 

    Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def analyze_power_bands(features: dict, labels: list, start_times: list, mouse_id: int, lab: str, labs_channels: dict):
    """writes given data to the passed table, each sample is written in a new row"""

    bands = {'slow_delta': [0.5, 2.25],
             'fast_delta': [2.5, 4],
             'slow_theta': [5, 8],
             'fast_theta': [8, 10],
             'alpha': [9, 14],
             'beta':  [14, 30]
            }

    recording_list = []

    # iterate over samples and create rows
    for sample_start, label in zip(start_times, labels):
        # determine idxs of data to load from features
        sample_end = int(sample_start + config.SAMPLE_DURATION * config.SAMPLING_RATE)

        for c in labs_channels[lab]:
            signal = features[c][sample_start:sample_end]

            epoch_dict = {
                'mouse_id': mouse_id,
                'lab': lab,
                'channel': c,
                'label': label
                }

            if 'EEG' in c:
            
                for b in bands:
                    rel_bp = bandpower(
                        data=signal,
                        sf=config.SAMPLING_RATE,
                        band=bands[b],
                        method='multitaper', 
                        relative=True)

                    band_dict = epoch_dict.copy()
                    band_dict['band'] = b
                    band_dict['power'] = rel_bp
                
                    recording_list.append(band_dict)

            # else:
            #     # Define window length (4 seconds)
            #     win = 4 * config.SAMPLING_RATE
            #     freqs, psd = welch(signal, config.SAMPLING_RATE, nperseg=win)
            #     freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
            #     total_power = simps(psd, dx=freq_res)

            #     band_dict = epoch_dict.copy()
            #     band_dict['band'] = 'total'
            #     band_dict['power'] = total_power
                
            #     recording_list.append(band_dict)

    return recording_list


def create_df_tuebingen(dataframe_list):
    stages = ['Wake', 'REM', 'Non REM']
    file = tables.open_file("/home/s202283/code/grieger_validation/cache/dataset/data_tuebingen_main_stages.h5")
    data = file.root['train']

    start = time.time()
    for i in range(len(data)):
        if i%10000==0 and i>0:
            logging.info("{} out of {} Tübingen rows processed".format(i, len(data)))
            logging.info('execution time: {:.2f}'.format(time.time() - start))
            start = time.time()

        row = data[i]

        mouse_id = row['mouse_id']
        feature = row['EEG_PR']
        label = str(row['label'], 'utf-8')

        bands = {
            'slow_delta': [0.5, 2.25],
            'fast_delta': [2.5, 4],
            'slow_theta': [5, 8],
            'fast_theta': [8, 10],
            'alpha': [9, 14],
            'beta':  [14, 30]
        }

        recording_list = []

        epoch_dict = {
            'mouse_id': mouse_id,
            'lab': 'Tuebingen',
            'channel': 'EEG_PR',
            'label': label
            }
            
        for b in bands:
            rel_bp = bandpower(
                data=feature,
                sf=64,
                band=bands[b],
                method='multitaper', 
                relative=True)

            band_dict = epoch_dict.copy()
            band_dict['band'] = b
            band_dict['power'] = rel_bp

            dataframe_list.append(band_dict)

    # transform the label to it's index in STAGES
    return dataframe_list


def create_df():
    """transform files in DATA_DIR to pytables table"""
    # load description of table columns
    table_desc = create_table_description(config)

    logging.info(f'data is loaded from {realpath(config.DATA_DIR)}')

    rows_df = []

    # determine which files to transform for each dataset based on DATA_SPLIT
    labs = [f for f in os.listdir(config.DATA_DIR) if not f.startswith('.') and 'MACOSX' not in f]
    # iterate over files, load them and write them to the created table
    for l in labs:
        lab_name = l.split('-')[0]

        mice = [f for f in os.listdir(os.path.join(config.DATA_DIR, l)) if not f.startswith('.') and 'MACOSX' not in f]

        for m in mice:
            try:
                assert os.path.isfile(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'signal.mat'))
            except:
                logging.info(os.path.join(l, m), ' does not exist')
                continue

            if os.path.isfile(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'fs.mat')):
                original_fs = scipy.io.loadmat(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'fs.mat'))['fs'][0][0]
            else:
                original_fs = config.ORIGINAL_FS[lab_name]
    
            logging.info('mouse {:s}'.format(os.path.join(l, m)))
            start = time.time()

            if lab_name == 'Kornum':
                labels_name = 'labels.csv'
            elif lab_name == 'Maiken':
                labels_name = 'newSS.mat'
            else:
                labels_name = 'labels.mat'

            features, times = read_recording(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'signal.mat'), original_fs, config.LABS_CHANNELS, lab_name)
            labels = read_labels(os.path.join(config.DATA_DIR, l, m, 'cleaned', labels_name))
            
            rows_df = rows_df + analyze_power_bands(features, labels, times, m, lab_name, config.LABS_CHANNELS)

            logging.info('execution time: {:.2f}'.format(time.time() - start))
    
    logging.info('starting tuebingen dataset')
    rows_df = create_df_tuebingen(rows_df)  

    df = pd.DataFrame(rows_df)
    df.to_csv('/home/s202283/code/grieger_validation/power_bands.csv', index=False)
    logging.info('dataframe computation finished')
    # fig = plt.figure(figsize=(11, 5.7))
    # bands_plot = sns.boxplot(data=df.loc[(df['channel']!='EMG') & (df['label']=='Wake')], x="lab", y="power", hue='band', palette=sns.color_palette("Set2"))  #, errorbar='sd'
    # plt.savefig('hola.png')
    # fig = plt.figure(figsize=(11, 5.7))
    # bands_plot = sns.barplot(data=df.loc[(df['channel']=='EMG')], x="lab", y="power", hue='label', palette=sns.color_palette("Set2"))  #, errorbar='sd'
    # plt.savefig('hola.png')


if __name__ == '__main__':
    args = parse()
    # load config, dirs are not needed here, because we do not write a log
    logging.basicConfig(level=logging.DEBUG)

    config = ConfigLoader(experiment=args.experiment, create_dirs=False)

    # transform files
    logging.info('dataframe computation started')
    create_df()