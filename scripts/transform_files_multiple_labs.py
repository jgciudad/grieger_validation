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
import pandas as pd

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


def write_data_to_table(table: tables.Table, features: dict, labels: list, start_times: list, mouse_id: int, lab: str, labs_channels: dict):
    """writes given data to the passed table, each sample is written in a new row"""
    sample = table.row

    # iterate over samples and create rows
    for sample_start, label in zip(start_times, labels):
        # determine idxs of data to load from features
        sample_end = int(sample_start + config.SAMPLE_DURATION * config.SAMPLING_RATE)

        # try to load data from sample_start to sample_end, if there is not enough data, ignore the sample
        try:
            sample[COLUMN_MOUSE_ID] = mouse_id
            sample[COLUMN_LAB] = lab
            for c in labs_channels[lab].keys():
                sample[c] = features[c][sample_start:sample_end]
            sample[COLUMN_LABEL] = label
            sample.append()
        except ValueError:
            print(f"""
            While processing epoch [{sample_start}, {sample_end}] with label {label}:
            not enough datapoints in file (n = {len(list(features.values())[0])})
            This epoch is ignored.
            """)
    # write data to table
    table.flush()


def transform():
    """transform files in DATA_DIR to pytables table"""
    # load description of table columns
    table_desc = create_table_description(config)

    # if the transformed data file already exists, ask the user if he wants to overwrite it
    if isfile(config.DATA_FILE):
        question = f"{realpath(config.DATA_FILE)} already exists, do you want to override? (y/N): "
        response = input(question)
        if response.lower() != 'y':
            exit()

    print(f'data is loaded from {realpath(config.DATA_DIR)}')

    # open pytables DATA_FILE
    with tables.open_file(config.DATA_FILE, mode='w', title='data from multiple labs') as f:

        # create tables for every dataset
        table = f.create_table(f.root, 'multiple_labs', table_desc, 'multiple_labs_data')

        # determine which files to transform for each dataset based on DATA_SPLIT
        labs = [f for f in os.listdir(config.DATA_DIR) if '.' not in f]
        # iterate over files, load them and write them to the created table
        for l in labs:
            lab_name = l.split('-')[0]

            mice = [f for f in os.listdir(os.path.join(config.DATA_DIR, l)) if not f.startswith('.')]

            for m in mice:
                try:
                    assert os.path.isfile(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'signal.mat'))
                except:
                    print(os.path.join(l, m), ' does not exist')
                    continue

                if os.path.isfile(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'fs.mat')):
                    original_fs = scipy.io.loadmat(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'fs.mat'))['fs'][0][0]
                else:
                    original_fs = config.ORIGINAL_FS[lab_name]
        
                print('mouse {:s}'.format(os.path.join(l, m)))
                start = time.time()

                if lab_name == 'Kornum':
                    labels_name = 'labels.csv'
                elif lab_name == 'Maiken':
                    labels_name = 'newSS.mat'
                else:
                    labels_name = 'labels.mat'

                features, times = read_recording(os.path.join(config.DATA_DIR, l, m, 'cleaned', 'signal.mat'), original_fs, config.LABS_CHANNELS, lab_name)
                labels = read_labels(os.path.join(config.DATA_DIR, l, m, 'cleaned', labels_name))

                # write loaded data to table
                write_data_to_table(table, features, labels, times, m, lab_name, config.LABS_CHANNELS)

                print('execution time: {:.2f}'.format(time.time() - start))
                print()

        print(f)


if __name__ == '__main__':
    args = parse()
    # load config, dirs are not needed here, because we do not write a log
    config = ConfigLoader(args.experiment, create_dirs=False)

    # transform files
    transform()
