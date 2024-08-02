#!/usr/bin/env python

import argparse
import sys
from importlib import import_module
from os.path import basename, join, dirname, realpath, isfile
import os
import torch
import torch.utils.data as t_data
import openpyxl
import sklearn.metrics

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
from base.logger import Logger
from base.data.dataloader import TuebingenDataloader
from base.evaluation.result_logger import ResultLogger




def parse():
    parser = argparse.ArgumentParser(description='evaluate exp')
    parser.add_argument('--experiment', '-e', required=True,
                        help='name of experiment to run')
    parser.add_argument('--test_lab', '-t', required=True,
                        help="lab to test the model on: 'Antoine', 'Kornum', 'Alessandro', 'Sebastian' or 'Maiken'")
    parser.add_argument('--save_dir', '-s', required=True,
                        help="path to save model'")                    

    return parser.parse_args()


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


def analyze_power_bands(features: dict, true_labels: list, predicted_labels: list, lab: str):
    """writes given data to the passed table, each sample is written in a new row"""
    
    labels_correspondence = {'1': 'Wake',
                            '2': 'Non REM',
                            '3': 'REM',
                            '4': 'Art'}

    bands = {'slow_delta': [0.5, 2.25],
             'fast_delta': [2.5, 4],
             'slow_theta': [5, 8],
             'fast_theta': [8, 10],
             'alpha': [9, 14],
             'beta':  [14, 30]
            }

    recording_list = []

    # iterate over samples and create rows
    for feature, true_label, predicted_label in zip(features, true_labels, predicted_labels):
        # determine idxs of data to load from features

        # for c in labs_channels[lab]:
        epoch_dict = {
            'lab': lab,
            'true_label': labels_correspondence[str(true_label+1)],
            'predicted_label': labels_correspondence[str(predicted_label+1)]
            }
        
        for b in bands:
            rel_bp = bandpower(
                data=np.squeeze(feature),
                sf=config.SAMPLING_RATE,
                band=bands[b],
                method='multitaper', 
                relative=False)

            band_dict = epoch_dict.copy()
            band_dict['band'] = b
            band_dict['power'] = rel_bp
        
            recording_list.append(band_dict)

    return recording_list


def evaluate_power_bands(test_lab):
    """evaluates best model in experiment on given dataset"""
    logger.fancy_log('start evaluation')
    result_logger = ResultLogger(config)

    # create dataloader for given dataset, the data should not be altered in any way
    map_loader = TuebingenDataloader(config, 'test', test_lab,
                                   data_fraction=config.DATA_FRACTION)
    dataloader = t_data.DataLoader(map_loader, batch_size=config.BATCH_SIZE_EVAL, shuffle=False, num_workers=4)

    # create empty model from model name in config and set it's state from best model in EXPERIMENT_DIR
    model = import_module('.' + config.MODEL_NAME, 'base.models').Model(config).to(config.DEVICE).eval()
    model_file = join(config.EXPERIMENT_DIR, config.MODEL_NAME + '-best.pth')
    if isfile(model_file):
        model.load_state_dict(torch.load(model_file)['state_dict'])
        # model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'))['state_dict'])
    else:
        raise ValueError('model_file {} does not exist'.format(model_file))
    logger.logger.info('loaded model:\n' + str(model))

    # evaluate model
    model = model.eval()
    predicted_labels = np.empty(0, dtype='int')
    actual_labels = np.empty(0, dtype='int')

    rows_df = []

    i=0
    with torch.no_grad():
        for data in dataloader:
            logger.logger.info("{} out of {} sleep epochs computed.".format(i*config.BATCH_SIZE_EVAL, len(dataloader)*config.BATCH_SIZE_EVAL))
            features, labels, labs = data
            features = features.to(config.DEVICE)
            labels = labels.long().to(config.DEVICE)

            outputs = model(features)

            _, predicted_labels_i = torch.max(outputs, dim=1)

            predicted_labels = np.r_[predicted_labels, predicted_labels_i.tolist()]
            actual_labels = np.r_[actual_labels, labels.tolist()]

            rows_df = rows_df + analyze_power_bands(features.cpu().numpy(), labels.cpu().numpy(), predicted_labels_i.cpu().numpy(), test_lab)

            i+=1
        
    result_logger.log_sleep_stage_f1_scores(actual_labels, predicted_labels, test_lab)
    logger.logger.info('')
    result_logger.log_confusion_matrix(actual_labels, predicted_labels, test_lab, wo_plot=False)
    result_logger.log_transformation_matrix(actual_labels, predicted_labels, test_lab,
                                            wo_plot=False)

    df = pd.DataFrame(rows_df)
    df.to_csv('/home/s202283/code/grieger_validation/abs_power_bands_' + test_lab + '.csv', index=False)

    
if __name__ == '__main__':
    args = parse()
    config = ConfigLoader(save_dir=os.path.join(args.save_dir, args.test_lab), experiment=args.experiment)

    logger = Logger(config)  # wrapper for logger
    logger.init_log_file(args, basename(__file__))  # create log file and log config, etc

    logger.fancy_log('evaluate best model of experiment {} on dataset {}'.format(args.experiment, args.test_lab))
    # perform evaluation
    evaluate_power_bands(test_lab=args.test_lab)