import collections
import time
from importlib import import_module
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np
import yaml
from torch import optim


def update_dict(d_to_update: dict, update: dict):
    """method to update a dict with the entries of another dict

    `d_to_update` is updated by `update`"""
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            d_to_update[k] = update_dict(d_to_update.get(k, {}), v)
        else:
            d_to_update[k] = v
    return d_to_update


class ConfigLoader:
    def __init__(self, experiment='standard_config', create_dirs=True):
        """general class to load config from yaml files into fields of an instance of this class

        first the config from standard_config.yml is loaded and then updated with the entries in the config file
        specified by `experiment`

        for a description of the various configurations see README.md

        Args:
            experiment (str): name of the config file to load without the .yml file extension; file must be in folder
                'config'
            create_dirs (bool): if set to False EXPERIMENT_DIR, MODELS_DIR and VISUALS_DIR are not created
        """
        base_dir = realpath(join(dirname(__file__), '..'))

        self.experiment = experiment
        config = self.load_config()

        self.RUN_NAME = experiment + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
        """identifier for the run of the experiment, used in log_file and `VISUALS_DIR`"""

        # general
        self.DEVICE = config['general']['device']
        assert self.DEVICE in ['cpu', 'cuda'], 'DEVICE only support `cpu` or `cuda`'

        # dirs
        self.EXPERIMENT_DIR = join(base_dir, 'results', self.experiment)
        """general directory with results from experiment"""
        self.MODELS_DIR = join(self.EXPERIMENT_DIR, 'models')
        """directory in `EXPERIMENT_DIR` for `extra_safe` models"""
        self.VISUALS_DIR = join(self.EXPERIMENT_DIR, 'visuals', self.RUN_NAME)
        """directory in `EXPERIMENT_DIR` for all generated plots"""
        if create_dirs:
            makedirs(self.EXPERIMENT_DIR, exist_ok=True)
            makedirs(self.MODELS_DIR, exist_ok=True)
            makedirs(self.VISUALS_DIR, exist_ok=True)

        self.DATA_DIR = realpath(config['dirs']['data'])
        cache_dir = config['dirs']['cache']
        makedirs(cache_dir, exist_ok=True)

        # data
        self.SAMPLE_DURATION = config['data']['sample_duration']
        assert type(self.SAMPLE_DURATION) in [int, float]
        self.SAMPLING_RATE = config['data']['sampling_rate']
        assert type(self.SAMPLE_DURATION) is int
        self.ORIGINAL_FS = config['data']['original_fs']

        # experiment
        exp_config = config['experiment']
        # data
        data_config = exp_config['data']
        self.DATA_FILE = join(cache_dir, data_config['file'])
        self.STAGES = data_config['stages']
        # self.MASK_ARTIFACTS = data_config['mask_artifacts']
        # self.BALANCED_TRAINING = data_config['balanced_training']
        # assert type(self.BALANCED_TRAINING) is bool
        # self.BALANCING_WEIGHTS = data_config['balancing_weights']
        # assert np.all([w >= 0 for w in self.BALANCING_WEIGHTS]), 'BALANCING_WEIGHTS must be non negative'
        self.CHANNELS_IN_MODEL = data_config['channels_in_model']
        self.CHANNELS_IN_TABLE = data_config['channels_in_table']
        self.LABS_CHANNELS = data_config['labs_channels']
        self.SAMPLES_LEFT = data_config['samples_left']
        # assert type(self.SAMPLES_LEFT) is int
        self.SAMPLES_RIGHT = data_config['samples_right']
        # assert type(self.SAMPLES_RIGHT) is int
        self.LABS = data_config['labs']
        self.ORIGINAL_DATASET_SIZE = data_config['original_dataset_size']
        self.VALIDATION_SPLIT = data_config['validation_split']

        # training
        training_config = exp_config['training']
        self.LOG_INTERVAL = training_config['log_interval']
        assert type(self.LOG_INTERVAL) is int
        self.EXTRA_SAFE_MODELS = training_config['additional_model_safe']
        assert type(self.EXTRA_SAFE_MODELS) is bool
        self.BATCH_SIZE = training_config['batch_size']
        assert type(self.BATCH_SIZE) is int
        self.DATA_FRACTION = training_config['data_fraction']
        assert type(self.DATA_FRACTION) is bool
        self.EPOCHS = training_config['epochs']
        assert type(self.EPOCHS) is int
        self.WARMUP_EPOCHS = training_config['optimizer']['scheduler']['warmup_epochs']
        assert type(self.WARMUP_EPOCHS) is int
        self.S_OPTIM_MODE = training_config['optimizer']['scheduler']['mode']
        assert self.S_OPTIM_MODE in ['step', 'exp', 'half', 'plat', None], \
            'S_OPTIM_MODE is not one of ["step", "exp", "half", "plat", None]'
        self.S_OPTIM_PARAS = training_config['optimizer']['scheduler']['parameters']
        assert type(self.S_OPTIM_PARAS) in [list, type(None)], 'S_OPTIM_PARAS must be a list or None'
        self.LEARNING_RATE = training_config['optimizer']['learning_rate']
        assert type(self.LEARNING_RATE) is float
        self.OPTIMIZER = getattr(optim, training_config['optimizer']['class'])
        assert issubclass(self.OPTIMIZER, optim.Optimizer), 'OPTIMIZER must be a subtype of optim.optimizer.Optimizer'
        self.OPTIM_PARAS = training_config['optimizer']['parameters']
        assert type(self.OPTIM_PARAS) is dict, 'OPTIM_PARAS must be a dict'
        self.L1_WEIGHT_DECAY = training_config['optimizer']['l1_weight_decay']
        assert type(self.L1_WEIGHT_DECAY) in [float, int]
        self.L2_WEIGHT_DECAY = training_config['optimizer']['l2_weight_decay']
        assert type(self.L2_WEIGHT_DECAY) in [float, int]

        # evaluation
        self.BATCH_SIZE_EVAL = config['experiment']['evaluation']['batch_size']
        assert type(self.BATCH_SIZE_EVAL) is int

        # model
        model_config = exp_config['model']
        self.FILTERS = model_config['filters']
        assert type(self.FILTERS) is int
        self.CLASSIFIER_DROPOUT = model_config['classifier_dropout']
        assert type(self.CLASSIFIER_DROPOUT) is list and np.all([type(o) is float for o in self.CLASSIFIER_DROPOUT])
        self.FEATURE_EXTR_DROPOUT = model_config['feature_extr_dropout']
        assert type(self.FEATURE_EXTR_DROPOUT) is list and np.all([type(o) is float for o in self.FEATURE_EXTR_DROPOUT])
        self.MODEL_NAME = model_config['name']
        try:
            module = import_module('.' + self.MODEL_NAME, 'base.models')
            assert getattr(module, 'Model') is not None
        except:
            assert False


    def load_config(self):
        """loads config from standard_config.yml and updates it with <experiment>.yml"""
        base_dir = realpath(join(dirname(__file__), '..'))
        with open(join(base_dir, 'config', 'standard_config_multiple_labs.yml'), 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
        if self.experiment != 'standard_config_multiple_labs':
            with open(join(base_dir, 'config', self.experiment + '.yml'), 'r') as ymlfile:
                config = update_dict(config, yaml.safe_load(ymlfile))
        return config
