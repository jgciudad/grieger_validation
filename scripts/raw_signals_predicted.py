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

from scipy.signal import butter
from scipy.signal import filtfilt, sosfiltfilt
from typing import Tuple

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


def save_signals(features: dict, true_labels: list, predicted_labels: list, lab: str, signals_array, bands_array, filtered_signals_array, filtered_bands_array, filtered_predictions, ground_truth_list, counter, sos, model):
    """writes given data to the passed table, each sample is written in a new row"""
    
    labels_correspondence = {'1': 'Wake',
                            '2': 'Non REM',
                            '3': 'REM',
                            '4': 'Art'}

    bands = {'slow_delta': [0, 2.5],
             'fast_delta': [2.5, 4],
             'slow_theta': [4, 8],
             'fast_theta': [8, 10],
             'alpha': [10, 14],
             'beta':  [14, 30]
            }

    # iterate over samples and create rows
    for feature, true_label, predicted_label in zip(features, true_labels, predicted_labels):
        # determine idxs of data to load from features

        if predicted_label == 0 and counter<15:
        # if counter<15:
            filtered_feature = sosfiltfilt(sos, feature, axis=-1, padtype='constant')
            # filtered_feature = filtfilt(sos[0], sos[1], feature, axis=-1, padtype='constant')
            # filtered_feature = feature*1.4

            signals_array[counter, :] = feature
            filtered_signals_array[counter, :] = filtered_feature

            filtered_output = model(torch.from_numpy(np.expand_dims(filtered_feature, 0).copy()).to(config.DEVICE).float())
            filtered_pred_class = torch.max(filtered_output, dim=1)[1].item()
            filtered_predictions.append(labels_correspondence[str(filtered_pred_class+1)])
            
            ground_truth_list.append(labels_correspondence[str(true_label+1)])


            for b_idx, b in enumerate(bands):
                rel_bp = bandpower(
                    data=np.squeeze(feature),
                    sf=config.SAMPLING_RATE,
                    band=bands[b],
                    method='multitaper', 
                    relative=True)

                bands_array[counter, b_idx] = rel_bp  

                filt_rel_bp = bandpower(
                    data=np.squeeze(filtered_feature),
                    sf=config.SAMPLING_RATE,
                    band=bands[b],
                    method='multitaper', 
                    relative=True)

                filtered_bands_array[counter, b_idx] = filt_rel_bp   

            counter += 1

        elif counter>=15:
            break

    return signals_array, filtered_signals_array, bands_array, filtered_bands_array, filtered_predictions, ground_truth_list, counter


def bandpass(sr: float, band: Tuple[float, float], outtype='sos', btype='bandpass') -> Tuple[np.ndarray, np.ndarray]:
    bandpass_freqs = 2 * np.array(band) / sr
    return butter(4, bandpass_freqs, btype=btype, output=outtype)


def save_predicted_signals(test_lab):
    """evaluates best model in experiment on given dataset"""
    logger.fancy_log('start evaluation')
    result_logger = ResultLogger(config)

    # create dataloader for given dataset, the data should not be altered in any way
    map_loader = TuebingenDataloader(config, 'test', test_lab,
                                   data_fraction=config.DATA_FRACTION)
    dataloader = t_data.DataLoader(map_loader, batch_size=config.BATCH_SIZE_EVAL, shuffle=True, num_workers=4)

    # create empty model from model name in config and set it's state from best model in EXPERIMENT_DIR
    model = import_module('.' + config.MODEL_NAME, 'base.models').Model(config).to(config.DEVICE).eval()
    model_file = join(config.EXPERIMENT_DIR, config.MODEL_NAME + '-best.pth')
    if isfile(model_file):
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'))['state_dict'])
        # model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'))['state_dict'])
    else:
        raise ValueError('model_file {} does not exist'.format(model_file))
    logger.logger.info('loaded model:\n' + str(model))

    # evaluate model
    model = model.eval()
    predicted_labels = np.empty(0, dtype='int')
    actual_labels = np.empty(0, dtype='int')

    rows_df = []

    counter = 0
    filtered_predictions = []
    ground_truth_list = []
    bands_array = np.zeros((15, 6))
    signals_array = np.zeros((15, 1920))
    filtered_bands_array = np.zeros((15, 6))
    filtered_signals_array = np.zeros((15, 1920))
    # sos = bandpass(64, (1, 0.4 * 64), 'sos', 'bandpass')
    sos = bandpass(64, 29, 'sos', 'lowpass')
    # sos = (np.array([1]), np.array([1, 0]))

    with torch.no_grad():
        for data in dataloader:
            if counter < 15:
                features, labels, labs = data
                features = features.to(config.DEVICE)
                labels = labels.long().to(config.DEVICE)

                outputs = model(features)

                _, predicted_labels_i = torch.max(outputs, dim=1)

                predicted_labels = np.r_[predicted_labels, predicted_labels_i.tolist()]
                actual_labels = np.r_[actual_labels, labels.tolist()]

                signals_array, filtered_signals_array, bands_array, filtered_bands_array, filtered_predictions, ground_truth_list, counter = save_signals(
                    features.cpu().numpy(),
                    labels.cpu().numpy(),
                    predicted_labels_i.cpu().numpy(),
                    test_lab,
                    signals_array,
                    bands_array,
                    filtered_signals_array,
                    filtered_bands_array,
                    filtered_predictions,
                    ground_truth_list,
                    counter,
                    sos,
                    model)

            else:
                break
    
    fig, ax = plt.subplots(3, 5, figsize=(19,9))
    for i in range(15):
        ax[i//5, i%5].plot(signals_array[i,:])
        ax[i//5, i%5].set_title("Real: {}, Pred: {}".format(ground_truth_list[i], 'Wake'))
    plt.tight_layout()
    plt.savefig('raw_wake_signals.jpg')

    fig, ax = plt.subplots(3, 5, figsize=(19,9))
    for i in range(15):
        ax[i//5, i%5].bar(x=['slow_delta', 'fast_delta', 'slow_theta', 'fast_theta', 'alpha', 'beta'], height=bands_array[i,:])
        ax[i//5, i%5].set_title("Real: {}, Pred: {}".format(ground_truth_list[i], 'Wake'))
        ax[i//5, i%5].set_ylim((0, 1))
    plt.tight_layout()
    plt.savefig('bands_wake_signals.jpg')

    fig, ax = plt.subplots(3, 5, figsize=(19,9))
    for i in range(15):
        ax[i//5, i%5].plot(filtered_signals_array[i,:])
        ax[i//5, i%5].set_title("Real: {}, Pred: {}".format(ground_truth_list[i], filtered_predictions[i]))
    plt.tight_layout()
    plt.savefig('filtered_wake_signals.jpg')

    fig, ax = plt.subplots(3, 5, figsize=(19,9))
    for i in range(15):
        ax[i//5, i%5].bar(x=['slow_delta', 'fast_delta', 'slow_theta', 'fast_theta', 'alpha', 'beta'], height=filtered_bands_array[i,:])
        ax[i//5, i%5].set_title("Real: {}, Pred: {}".format(ground_truth_list[i], filtered_predictions[i]))
        ax[i//5, i%5].set_ylim((0, 1))
    plt.tight_layout()
    plt.savefig('filtered_bands_wake_signals.jpg')

    fig, ax = plt.subplots(3, 5, figsize=(19,9))#, sharex=True, sharey=True)
    for i in range(15):
        N = signals_array[i,:].shape[0] #number of elements
        t = np.linspace(0, N * 3600, N)

        fft = np.fft.fft(signals_array[i,:])

        T = t[1] - t[0]

        P2 = abs(fft/N)
        P1 = P2[0:int(N/2 + 1)]
        P1[1:-1] = 2*P1[1:-1]

        f = 64/N*np.linspace(0,int(N/2), int(N/2)+1)
        ax[i//5, i%5].set_ylabel("Amplitude")
        ax[i//5, i%5].set_xlabel("Frequency [Hz]")
        ax[i//5, i%5].plot(f,P1)
        ax[i//5, i%5].set_xticks(np.linspace(0, 32, 9))
        ax[i//5, i%5].set_title("Real: {}, Pred: {}".format(ground_truth_list[i], 'Wake'))
    plt.tight_layout()
    plt.savefig('fourier_raw.jpg')

    fig, ax = plt.subplots(3, 5, figsize=(19,9))#, sharex=True, sharey=True)
    for i in range(15):
        N = signals_array[i,:].shape[0] #number of elements
        t = np.linspace(0, N * 3600, N)

        fft = np.fft.fft(signals_array[i,:])

        T = t[1] - t[0]

        P2 = abs(fft/N)
        P1 = P2[0:int(N/2 + 1)]
        P1[1:-1] = 2*P1[1:-1]

        f = 64/N*np.linspace(0,int(N/2), int(N/2)+1)
        ax[i//5, i%5].set_ylabel("Amplitude")
        ax[i//5, i%5].set_xlabel("Frequency [Hz]")
        ax[i//5, i%5].plot(f[:int(len(P1)/8)],P1[:int(len(P1)/8)])
        ax[i//5, i%5].set_xticks(np.linspace(0, 4, 9))
        ax[i//5, i%5].set_title("Real: {}, Pred: {}".format(ground_truth_list[i], 'Wake'))
    plt.tight_layout()
    plt.savefig('fourier_raw2.jpg')

    fig, ax = plt.subplots(3, 5, figsize=(19,9))#, sharex=True, sharey=True)
    for i in range(15):
        N = signals_array[i,:].shape[0] #number of elements
        t = np.linspace(0, N * 3600, N)

        fft = np.fft.fft(filtered_signals_array[i,:])

        T = t[1] - t[0]

        P2 = abs(fft/N)
        P1 = P2[0:int(N/2 + 1)]
        P1[1:-1] = 2*P1[1:-1]

        f = 64/N*np.linspace(0,int(N/2), int(N/2)+1)
        ax[i//5, i%5].set_ylabel("Amplitude")
        ax[i//5, i%5].set_xlabel("Frequency [Hz]")
        ax[i//5, i%5].plot(f,P1)
        ax[i//5, i%5].set_xticks(np.linspace(0, 32, 9))
        ax[i//5, i%5].set_title("Real: {}, Pred: {}".format(ground_truth_list[i], filtered_predictions[i]))
    plt.tight_layout()
    plt.savefig('filtered_fourier_raw.jpg')

    fig, ax = plt.subplots(3, 5, figsize=(19,9))#, sharex=True, sharey=True)
    for i in range(15):
        N = signals_array[i,:].shape[0] #number of elements
        t = np.linspace(0, N * 3600, N)

        fft = np.fft.fft(filtered_signals_array[i,:])

        T = t[1] - t[0]

        P2 = abs(fft/N)
        P1 = P2[0:int(N/2 + 1)]
        P1[1:-1] = 2*P1[1:-1]

        f = 64/N*np.linspace(0,int(N/2), int(N/2)+1)
        ax[i//5, i%5].set_ylabel("Amplitude")
        ax[i//5, i%5].set_xlabel("Frequency [Hz]")
        ax[i//5, i%5].plot(f[:int(len(P1)/8)],P1[:int(len(P1)/8)])
        ax[i//5, i%5].set_xticks(np.linspace(0, 4, 9))
        ax[i//5, i%5].set_title("Real: {}, Pred: {}".format(ground_truth_list[i], filtered_predictions[i]))
    plt.tight_layout()
    plt.savefig('filtered_fourier_raw2.jpg')

    plt.figure()
    w, h = scipy.signal.sosfreqz(sos, fs=64)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(np.linspace(0, 32, len(w)), db)
    plt.xlabel('Frequency (Hz)')
    plt.savefig('filter.png')

    print("FINISHED")
    
if __name__ == '__main__':
    args = parse()
    config = ConfigLoader(save_dir=os.path.join(args.save_dir, args.test_lab), experiment=args.experiment)

    logger = Logger(config)  # wrapper for logger
    logger.init_log_file(args, basename(__file__))  # create log file and log config, etc

    logger.fancy_log('evaluate best model of experiment {} on dataset {}'.format(args.experiment, args.test_lab))
    # perform evaluation
    save_predicted_signals(test_lab=args.test_lab)