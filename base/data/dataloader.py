import logging

import numpy as np
import random
import tables
import torch.utils.data as tud

from base.config_loader import ConfigLoader
from base.data.data_table import COLUMN_MOUSE_ID, COLUMN_LABEL, COLUMN_LAB
from base.utilities import format_dictionary

logger = logging.getLogger('tuebingen')


class TuebingenDataloader(tud.Dataset):
    """ dataloader for data from Tuebingen, requires the data to be stored in a pytables table with the structure
    described in `data_table.py`

    Each row in the table contains the data and label for one sample, without SAMPLES_LEFT and SAMPLES_RIGHT. This
    means, that every sample can be identified by it's index in the table, which is used during rebalancing. """

    def __init__(self, config, data_split, test_lab, balanced=False, data_fraction=False):
        """
        Args:
             config (ConfigLoader): config of the running experiment
             dataset (str): dataset from which to load data, must be a table in the pytables file
             balanced (bool): flag, whether the loaded data should be rebalanced by using BALANCING_WEIGHTS
        """
        self.config = config
        self.set = data_split
        self.balanced = balanced
        self.data_fraction = data_fraction
        self.data = None
        self.max_idx = 0
        self.test_lab = test_lab

        # file has to be opened here, so the indices for each stage can be loaded
        self.file = tables.open_file(self.config.DATA_FILE)
        self.labs_and_stages = self.get_lab_and_stage_data()

        if self.set == 'train':
            self.train_validation_split()
            self.train_indices, self.train_dist = self.get_indices(self.labs_and_stages_train)
            self.val_indices, self.val_dist = self.get_indices(self.labs_and_stages_val)
            self.loss_weights = self.get_loss_weights()

            self.train_dataloader = TuebingenDataLoaderSet(indices=self.train_indices, config=config, max_idx=self.max_idx, loss_weigths=self.loss_weights)
            self.val_dataloader = TuebingenDataLoaderSet(indices=self.val_indices, config=config, max_idx=self.max_idx)
        else:
            self.indices, _ = self.get_indices(self.labs_and_stages)

        self.file.close()

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here
        if self.data is None:  # open in thread
            self.file = tables.open_file(self.config.DATA_FILE)
            self.data = self.file.root['multiple_labs']

        # load internal index for rebalancing purposes, see get_indices()
        index = self.indices[index]

        # check that neighbors are within limits
        if isinstance(self.config.SAMPLES_LEFT, int) and isinstance(self.config.SAMPLES_RIGHT, int):
            left = int(self.config.SAMPLES_LEFT)
            right = int(self.config.SAMPLES_RIGHT)
        else:
            # load one additional sample to each side if decimal number of neigbors
            left = int(self.config.SAMPLES_LEFT) + 1
            right = int(self.config.SAMPLES_RIGHT) + 1
            left_rest = self.config.SAMPLES_LEFT % 1
            right_rest = self.config.SAMPLES_RIGHT % 1
        # calculate start and end to prevent IndexErrors
        idx_from = 0 if index - left < 0 else index - left
        idx_to = idx_from + left + right
        if idx_to >= self.max_idx:
            idx_from = self.max_idx - left - right - 1
            idx_to = idx_from + left + right
        index = int((idx_from + idx_to) / 2)

        # if the samples in the block are not from the same mouse, the block has to be shifted
        if not np.all(self.data[index][COLUMN_MOUSE_ID] == self.data[idx_from:idx_to + 1][COLUMN_MOUSE_ID]):
            idx = np.where(self.data[index][COLUMN_MOUSE_ID] != self.data[idx_from:idx_to + 1][COLUMN_MOUSE_ID])[0][0]
            dist_from_limits = min(idx, idx_to - idx_from - idx)  # how much to shift
            if dist_from_limits == idx:  # sample from wrong mouse is on the left side --> shift to the right
                idx_shift = dist_from_limits
            else:  # sample from wrong mouse is on the right side --> shift to the left
                idx_shift = -dist_from_limits
            idx_from += idx_shift
            idx_to += idx_shift
            index += idx_shift

        # sample EEG channels if the model takes as input less EEG channels than available
        channels_to_load = self.select_channels(index)

        # load the data specified by SAMPLES_LEFT and SAMPLES_RIGHT
        rows = self.data[idx_from:idx_to+1]
        feature = np.c_[[rows[c].flatten() for c in channels_to_load]]

        # crop edges if number of neighbors is a float
        if isinstance(self.config.SAMPLES_LEFT, float) and isinstance(self.config.SAMPLES_RIGHT, float):
            feature = feature[:, int(self.config.SAMPLING_RATE*self.config.SAMPLE_DURATION*(1-left_rest)) : -int(self.config.SAMPLING_RATE*self.config.SAMPLE_DURATION*(1-right_rest))]

        # transform the label to it's index in STAGES
        return feature, self.config.STAGES.index(str(self.data[index][COLUMN_LABEL], 'utf-8')), self.config.LABS.index(str(self.data[index][COLUMN_LAB], 'utf-8')), str(self.data[index]['mouse_id'], 'utf-8')

    def __len__(self):
        return self.indices.size
    
    def get_loss_weights(self):
        loss_weights = {}
        
        lab_sizes = [sum(self.train_dist[lab].values()) for lab in self.train_dist]

        # weights for both sleep stages and labs

        for lab in self.train_dist:
            loss_weights[lab] = {}

            for stage in self.config.STAGES:
                loss_weights[lab][stage] = sum(lab_sizes) / len(self.train_dist) / (len(self.config.STAGES)) / self.train_dist[lab][stage]
                        
        return loss_weights
    
    def select_channels(self, index):
        lab = self.data[index]['lab']
        eeg_channels_available = [x for x in list(self.config.LABS_CHANNELS[lab.decode('UTF-8')].keys()) if 'EEG' in x]

        n_eeg_channels_to_load = len([x for x in self.config.CHANNELS_IN_MODEL if 'EEG' in x])

        channels_to_load = []
        if len(eeg_channels_available) > n_eeg_channels_to_load:
            while len(channels_to_load) < n_eeg_channels_to_load:
                channels_to_load.append(eeg_channels_available.pop(random.randint(0,len(eeg_channels_available)-1)))
        elif len(eeg_channels_available) < n_eeg_channels_to_load:
            raise Exception("Passing more EEG channels than present in the data is not supported")
        else:
            channels_to_load = eeg_channels_available
        
        if 'EMG' in self.config.CHANNELS_IN_MODEL:
            channels_to_load.append('EMG')
        
        return channels_to_load

    def get_indices(self, labs_and_stages_set):
        """ loads indices of samples in the pytables table the dataloader returns

        if flag `balanced` is set, rebalancing is done here by randomly drawing samples from all samples in a stage
        until nitems * BALANCING_WEIGHTS[stage] is reached

        drawing of the samples is done with replacement, so samples can occur more than once in the dataloader """
        indices = np.empty(0)
        
        data_dist = {}
        for lab in labs_and_stages_set:
            data_dist[lab] = {}
            for stage in labs_and_stages_set[lab]:
                data_dist[lab][stage] = labs_and_stages_set[lab][stage].size
        logger.info(
            'data distribution in database for dataset ' + str(self.set) + ':\n' + format_dictionary(data_dist))

        # # apply balancing
        # if self.balanced:
        #     # the balancing weights are normed
        #     # if a stage has no samples, the weight belonging to it is set to 0
        #     balancing_weights = np.array(self.config.BALANCING_WEIGHTS, dtype='float')
        #     for n, stage_data in enumerate(self.stages):
        #         if len(stage_data) == 0:
        #             print('label ' + self.config.STAGES[n] + ' has zero samples')
        #             balancing_weights[n] = 0
        #     balancing_weights /= sum(balancing_weights)

        #     # draw samples according to balancing weights
        #     for n, z in enumerate(zip(self.stages, self.config.STAGES)):
        #         stage_data, stage = z
        #         if len(stage_data) == 0:
        #             continue
        #         indices = np.r_[indices, np.random.choice(stage_data, size=int(
        #             self.nitems * balancing_weights[n]) + 1, replace=True)].astype('int')
        #         self.data_dist[stage] = int(self.nitems * balancing_weights[n]) + 1
        #     np.random.shuffle(indices)  # shuffle indices, otherwise they would be ordered by stage...
        # else:  # if 'balanced' is not set, all samples are loaded
        for lab in labs_and_stages_set:
            for stage in labs_and_stages_set[lab]:
                indices = np.r_[indices, labs_and_stages_set[lab][stage]].astype('int')
        indices = np.sort(indices)  # the samples are sorted by index for the creation of a transformation matrix

        logger.info('data distribution after processing:\n' + format_dictionary(data_dist))

        return indices, data_dist

    def reset_indices(self):
        """ reload indices, only relevant for balancing purposes, because the samples are redrawn """
        self.indices = self.get_indices()
    
    def get_lab_and_stage_data(self):
        """ load indices of samples in the pytables table for each lab

        if data_fraction is set, load only a random fraction of the indices

        Returns:
            list: list with entries for each lab containing lists with indices of samples in that lab
        """
        lab_and_stage_data = {}
        table = self.file.root['multiple_labs']

        if self.set != 'test':
            for lab in self.config.LABS:
                if lab != self.test_lab:
                    lab_and_stage_data[lab] = {}
                    for stage in self.config.STAGES:
                        lab_and_stage_data[lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, lab, COLUMN_LABEL, stage))
                        if lab_and_stage_data[lab][stage].size > 0:
                            if max(lab_and_stage_data[lab][stage]) > self.max_idx:
                                self.max_idx = max(lab_and_stage_data[lab][stage])
        else:
            lab_and_stage_data[self.test_lab] = {}
            for stage in self.config.STAGES:
                lab_and_stage_data[self.test_lab][stage] = table.get_where_list('({}=="{}") & ({}=="{}")'.format(COLUMN_LAB, self.test_lab, COLUMN_LABEL, stage))
                if lab_and_stage_data[self.test_lab][stage].size > 0:
                    if max(lab_and_stage_data[self.test_lab][stage]) > self.max_idx:
                        self.max_idx = max(lab_and_stage_data[self.test_lab][stage])

        return lab_and_stage_data
    
    def train_validation_split(self):
        self.labs_and_stages_train = {}
        self.labs_and_stages_val = {}

        if self.config.DATA_FRACTION == True:
            num_samples_per_lab_train = int(self.config.ORIGINAL_DATASET_SIZE / len(self.labs_and_stages))
            num_samples_per_lab_val = int(num_samples_per_lab_train * self.config.VALIDATION_SPLIT)

            for lab in self.labs_and_stages:
                self.labs_and_stages_train[lab] = {}
                self.labs_and_stages_val[lab] = {}
                l_size = sum([s.size for s in self.labs_and_stages[lab].values()])

                for stage in self.labs_and_stages[lab]:
                    lab_stage_size = self.labs_and_stages[lab][stage].size
                    stage_ratio = lab_stage_size / l_size
                    
                    shuffled_indexes = np.random.permutation(self.labs_and_stages[lab][stage])
                    self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:-int(num_samples_per_lab_val*stage_ratio)])
                    self.labs_and_stages_val[lab][stage] = np.sort(shuffled_indexes[-int(num_samples_per_lab_val*stage_ratio):])

                    train_lab_stage_size = self.labs_and_stages_train[lab][stage].size

                    if train_lab_stage_size > num_samples_per_lab_train*stage_ratio:
                        l_downsampled = np.random.choice(self.labs_and_stages_train[lab][stage], size=int(num_samples_per_lab_train*stage_ratio), replace=False)
                        self.labs_and_stages_train[lab][stage] = l_downsampled
                    else:
                        l_upsampled = np.random.choice(self.labs_and_stages_train[lab][stage], size=int(num_samples_per_lab_train*stage_ratio), replace=True)
                        self.labs_and_stages_train[lab][stage] = l_upsampled

        else:

            for lab in self.labs_and_stages:
                self.labs_and_stages_train[lab] = {}
                self.labs_and_stages_val[lab] = {}

                for stage in self.labs_and_stages[lab]:
                    lab_stage_size = self.labs_and_stages[lab][stage].size
                    
                    shuffled_indexes = np.random.permutation(self.labs_and_stages[lab][stage])
                    self.labs_and_stages_train[lab][stage] = np.sort(shuffled_indexes[:-int(lab_stage_size * self.config.VALIDATION_SPLIT)])
                    self.labs_and_stages_val[lab][stage] = np.sort(shuffled_indexes[-int(lab_stage_size * self.config.VALIDATION_SPLIT):])


    

class TuebingenDataLoaderSet(TuebingenDataloader):

    def __init__(self, indices, config, max_idx, loss_weigths=None):
        self.indices = indices
        self.config = config
        self.file = tables.open_file(self.config.DATA_FILE)
        self.file.close()
        self.data = None
        self.max_idx = max_idx
        self.loss_weights = loss_weigths

