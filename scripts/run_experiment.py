#! /usr/bin/env python

import argparse
import sys
import time
from importlib import import_module
from os.path import basename, realpath, join, dirname

import torch.utils.data as t_data
from sklearn.metrics import f1_score

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.config_loader import ConfigLoader
from base.data.dataloader import TuebingenDataloader
from base.evaluation.evaluate_model import evaluate
from base.evaluation.result_logger import ResultLogger
from base.logger import Logger
from base.training.scheduled_optim import ScheduledOptim
from base.training.train_model import train, snapshot

import json

def parse():
    """define and parse arguments for the script"""
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--experiment', '-e', required=True,
                        help='name of experiment to run')

    return parser.parse_args()


def training():
    """train experiment as it is described in config"""
    result_logger = ResultLogger(config)  # wrapper for various methods to log/plot results

    # train dataloader with configured data augmentation and rebalancing
    dl_train = TuebingenDataloader(config, 'train', config.BALANCED_TRAINING, augment_data=True,
                                   data_fraction=config.DATA_FRACTION)
    # validation dataloader without modification of loaded data
    dl_valid = TuebingenDataloader(config, 'valid', False, augment_data=False)
    # multithreaded pytorch dataloaders with 4 workers each, train data is shuffled
    trainloader = t_data.DataLoader(dl_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    validationloader = t_data.DataLoader(dl_valid, batch_size=config.BATCH_SIZE_EVAL, shuffle=False, num_workers=4)

    # create model from model_name given in config and load it onto configured DEVICE
    model = import_module('.' + config.MODEL_NAME, 'base.models').Model(config).to(config.DEVICE)
    logger.logger.info('classifier:\n' + str(model))
    # scheduled optimizer containing the pytorch optimizer with parameters from config
    optimizer = ScheduledOptim(
        config.OPTIMIZER(filter(lambda p: p.requires_grad, model.parameters()),
                         weight_decay=config.L2_WEIGHT_DECAY, **config.OPTIM_PARAS),
        peak_lr=config.LEARNING_RATE, warmup_epochs=config.WARMUP_EPOCHS, total_epochs=config.EPOCHS,
        parameters=config.S_OPTIM_PARAS, mode=config.S_OPTIM_MODE)

    # save best results for early stopping and snapshot creation of best model
    best_epoch = 0
    best_avg_f1_score = 0
    # save metrics after each epoch for plots
    f1_scores = {'train': {stage: [] for stage in config.STAGES + ['avg']},
                 'valid': {stage: [] for stage in config.STAGES + ['avg']}}
    losses = {'train': [], 'valid': []}

    # iterate over EPOCHS, start with epoch 1
    for epoch in range(1, config.EPOCHS + 1):
        start = time.time()  # measure time each epoch takes
        if epoch > 1:
            dl_train.reset_indices()  # rebalance samples for each epoch

        # train epoch and save metrics
        labels_train, loss_train = train(config, epoch, model, optimizer, trainloader, loss_weigths=dl_train.loss_weights)
        losses['train'].append(loss_train)

        # evaluate epoch and save metrics
        labels_valid, loss_valid = evaluate(config, model, validationloader)
        losses['valid'].append(loss_valid)

        # calculate f1-scores for given validation and training labels and log them
        logger.logger.info('')
        f1_scores_train = result_logger.log_sleep_stage_f1_scores(labels_train['actual'], labels_train['predicted'],
                                                                  'train')
        for stage in f1_scores_train:
            f1_scores['train'][stage].append(f1_scores_train[stage])

        f1_scores_valid = result_logger.log_sleep_stage_f1_scores(labels_valid['actual'], labels_valid['predicted'],
                                                                  'valid')
        for stage in f1_scores_valid:
            f1_scores['valid'][stage].append(f1_scores_valid[stage])

        # model from the current epoch better than best model?
        new_best_model = f1_scores_valid['avg'] > best_avg_f1_score
        # log/plot confusion and transformation matrices
        result_logger.log_confusion_matrix(labels_train['actual'], labels_train['predicted'], 'train', wo_plot=True)
        result_logger.log_confusion_matrix(labels_valid['actual'], labels_valid['predicted'], 'valid',
                                           wo_plot=not new_best_model)
        result_logger.log_transformation_matrix(labels_valid['actual'], labels_valid['predicted'], 'valid',
                                                wo_plot=not new_best_model)

        # save model if it updates the best model
        if new_best_model:
            best_avg_f1_score = f1_scores_valid['avg']
            best_epoch = epoch
            snapshot(config, {
                'model': config.MODEL_NAME,
                'epoch': epoch,
                'validation_avg_f1_score': f1_scores_valid['avg'],
                'state_dict': model.state_dict(),
                'clas_optimizer': optimizer.state_dict(),
            }, config.EXTRA_SAFE_MODELS)

        end = time.time()
        logger.logger.info('[epoch {:3d}] execution time: {:.2f}s\t avg f1-score: {:.4f}\n'.format(
            epoch, (end - start), f1_scores_valid['avg']))

        # increase epoch in scheduled optimizer to update the learning rate
        optimizer.inc_epoch()

        # early stopping
        # stop training if the validation f1 score has not increased over the last 5 epochs
        # but only do so after WARMUP_EPOCHS was reached
        if epoch >= config.WARMUP_EPOCHS and epoch - best_epoch > 5:
            break

    # log/plot f1-score course and metrics over all epochs for both datasets
    result_logger.plot_f1_score_course(f1_scores)
    result_logger.log_metrics({'loss': losses})
    logger.fancy_log('finished training')
    logger.fancy_log('best model on epoch: {} \tf1-score: {:.4f}'.format(best_epoch, best_avg_f1_score))

    with open(join(config.VISUALS_DIR, "f1_score_curve.json"), 'w') as f:
        # indent=2 is not needed but makes the file human-readable 
        # if the data is nested
        json.dump(f1_scores['valid']['avg'], f, indent=2) 


if __name__ == '__main__':
    args = parse()
    config = ConfigLoader(args.experiment)  # load config from experiment

    logger = Logger(config)  # create wrapper for logger
    # create log_file and initialize it with the script arguments and the config
    logger.init_log_file(args, basename(__file__))

    logger.fancy_log('start training with model: {}'.format(config.MODEL_NAME))
    training()
