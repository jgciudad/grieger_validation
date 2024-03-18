import logging

import numpy as np
import random
import tables
import torch.utils.data as tud

from base.config_loader import ConfigLoader
from base.data.data_augmentor import DataAugmentor
from base.data.data_table import COLUMN_MOUSE_ID, COLUMN_LABEL, COLUMN_LAB
from base.utilities import format_dictionary

logger = logging.getLogger('tuebingen')


class TuebingenDataloader(tud.Dataset):
    """ dataloader for data from Tuebingen, requires the data to be stored in a pytables table with the structure
    described in `data_table.py`

    Each row in the table contains the data and label for one sample, without SAMPLES_LEFT and SAMPLES_RIGHT. This
    means, that every sample can be identified by it's index in the table, which is used during rebalancing. """

    def __init__(self, config, set, test_lab, balanced=False, augment_data=False, data_fraction=False):
        """
        Args:
             config (ConfigLoader): config of the running experiment
             dataset (str): dataset from which to load data, must be a table in the pytables file
             balanced (bool): flag, whether the loaded data should be rebalanced by using BALANCING_WEIGHTS
             augment_data (bool): flag, whether the data is to be augmented, see `DataAugmentor`
        """
        self.config = config
        self.augment_data = augment_data
        self.set = set
        self.balanced = balanced
        self.data_fraction = data_fraction
        self.data = None
        self.data_augmentor = DataAugmentor(config)
        self.max_idx = 0
        self.test_lab = test_lab

        # file has to be opened here, so the indices for each stage can be loaded
        self.file = tables.open_file(self.config.DATA_FILE)
        self.labs = self.get_lab_data()
        # max index is needed to calculate limits for the additional samples loaded by SAMPLES_LEFT and SAMPLES_RIGHT
        self.nitems = sum([len(l) for l in self.labs])
        self.indices = self.get_indices()

        if self.set == 'train':
            shuffled_indices = np.random.permutation(self.indices)
            self.train_dataloader = TuebingenDataLoaderSet(indices=np.sort(shuffled_indices[:int(self.indices.size*0.8)]), config=config, max_idx=self.max_idx, augment_data=self.augment_data, loss_weigths=self.loss_weights)
            self.val_dataloader = TuebingenDataLoaderSet(indices=np.sort(shuffled_indices[int(self.indices.size*0.8):]), config=config, max_idx=self.max_idx, augment_data=self.augment_data)

        self.file.close()

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here
        if self.data is None:  # open in thread
            self.file = tables.open_file(self.config.DATA_FILE)
            self.data = self.file.root['multiple_labs']

        # load internal index for rebalancing purposes, see get_indices()
        index = self.indices[index]

        # load one additional sample to each side for window warping and time shift
        left = self.config.SAMPLES_LEFT + 1
        right = self.config.SAMPLES_RIGHT + 1
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

        # load only the data specified by SAMPLES_LEFT and SAMPLES_RIGHT w/o the samples for window warping
        rows = self.data[idx_from + 1:idx_to]
        feature = np.c_[[rows[c].flatten() for c in channels_to_load]]

        # load samples to the left and right and use them for data augmentation
        if self.augment_data:
            sample_left = np.c_[[self.data[idx_from][c].flatten() for c in channels_to_load]]
            sample_right = np.c_[[self.data[idx_to][c].flatten() for c in channels_to_load]]
            feature = self.data_augmentor.alternate_signals(feature, sample_left, sample_right)

        # transform the label to it's index in STAGES
        return feature, self.config.STAGES.index(str(self.data[index][COLUMN_LABEL], 'utf-8'))

    def __len__(self):
        return self.indices.size
    
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

    def get_indices(self):
        """ loads indices of samples in the pytables table the dataloader returns

        if flag `balanced` is set, rebalancing is done here by randomly drawing samples from all samples in a stage
        until nitems * BALANCING_WEIGHTS[stage] is reached

        drawing of the samples is done with replacement, so samples can occur more than once in the dataloader """
        indices = np.empty(0)

        data_dist = {l: len(n) for l, n in zip([x for x in self.config.LABS if x != self.test_lab], self.labs)}
        logger.info(
            'data distribution in database for dataset ' + str(self.set) + ':\n' + format_dictionary(data_dist))

        # apply balancing
        if self.balanced:
            # the balancing weights are normed
            # if a stage has no samples, the weight belonging to it is set to 0
            balancing_weights = np.array(self.config.BALANCING_WEIGHTS, dtype='float')
            for n, stage_data in enumerate(self.stages):
                if len(stage_data) == 0:
                    print('label ' + self.config.STAGES[n] + ' has zero samples')
                    balancing_weights[n] = 0
            balancing_weights /= sum(balancing_weights)

            # draw samples according to balancing weights
            for n, z in enumerate(zip(self.stages, self.config.STAGES)):
                stage_data, stage = z
                if len(stage_data) == 0:
                    continue
                indices = np.r_[indices, np.random.choice(stage_data, size=int(
                    self.nitems * balancing_weights[n]) + 1, replace=True)].astype('int')
                data_dist[stage] = int(self.nitems * balancing_weights[n]) + 1
            np.random.shuffle(indices)  # shuffle indices, otherwise they would be ordered by stage...
        else:  # if 'balanced' is not set, all samples are loaded
            for lab_data in self.labs:
                indices = np.r_[indices, lab_data].astype('int')
            indices = np.sort(indices)  # the samples are sorted by index for the creation of a transformation matrix

        logger.info('data distribution after processing:\n' + format_dictionary(data_dist))

        return indices

    def reset_indices(self):
        """ reload indices, only relevant for balancing purposes, because the samples are redrawn """
        self.indices = self.get_indices()
    
    def get_lab_data(self):
        """ load indices of samples in the pytables table for each lab

        if data_fraction is set, load only a random fraction of the indices

        Returns:
            list: list with entries for each lab containing lists with indices of samples in that lab
        """
        labs = []
        table = self.file.root['multiple_labs']

        if self.set != 'test':
            for lab in self.config.LABS:
                if lab != self.test_lab:
                    labs.append(table.get_where_list('({}=="{}")'.format(COLUMN_LAB, lab)))
        else:
            labs.append(table.get_where_list('({}=="{}")'.format(COLUMN_LAB, self.test_lab)))

        self.max_idx = max([max(l) for l in labs if len(l)>0])

        if self.config.DATA_FRACTION == True and self.set != 'test':
            reduced_labs = []
            num_samples = int(self.config.ORIGINAL_DATASET_SIZE / len(labs))

            for l in labs:
                if l.size > num_samples:
                    l_downsampled = np.random.choice(l, size=num_samples, replace=False)
                    reduced_labs.append(l_downsampled)
                else:
                    l_upsampled = np.random.choice(l, size=num_samples, replace=True)
                    reduced_labs.append(l_upsampled)

            labs = reduced_labs

        return labs
    

class TuebingenDataLoaderSet(TuebingenDataloader):

    def __init__(self, indices, config, max_idx, augment_data):
        self.indices = indices
        self.config = config
        self.file = tables.open_file(self.config.DATA_FILE)
        self.file.close()
        self.data = None
        self.max_idx = max_idx
        self.augment_data = augment_data

