import argparse
import collections
import os
import time
import warnings
from functools import partial
from math import ceil

import numpy as np
import torch
import transformers

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import trainer as trainer_arch
import wandb
from parse_config import ConfigParser
import random


def main(config):
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    warnings.filterwarnings('ignore')
    logger = config.get_logger('train')
    wandb.init(project='QEN', name=config['exp_name'])
    wandb.config.update({arg: config['trainer'][arg] for arg in
                         ['lp', 'lc', 'epochs', 'test_batch_size', 'early_stop', 'accumulation_iters']})
    wandb.config.update(config['arch']['args'])
    wandb.config.update(config['train_data_loader']['args'])
    wandb.config.update(config['optimizer']['args'])
    wandb.config.update({'optimizer': config['optimizer']['type']})

    # setup data_loader instances
    print('initializing dataloader...')
    train_data_loader = config.initialize('train_data_loader', module_data, config['data_path'])
    logger.info(train_data_loader)

    # build model architecture, then print to console
    print('initializing model...')
    model = config.initialize('arch', module_arch, train_data_loader.dataset)
    # logger.info(model)

    # get function handles of loss and metrics
    print('initializing loss and metrics...')
    loss = getattr(module_loss, config['loss'])
    if config['loss'].startswith("FocalLoss"):
        loss = loss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    if config['loss'].startswith("info_nce") or config['loss'].startswith("bce_loss"):
        pre_metric = partial(module_metric.obtain_ranks, mode=1)  # info_nce_loss
    else:
        pre_metric = partial(module_metric.obtain_ranks, mode=0)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    print('initializing optimizer...')
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    if config['lr_scheduler']['library'] == 'torch':
        lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    elif 'warmup' in config['lr_scheduler']['type']:
        total_steps = int(
            ceil(len(train_data_loader) / config['trainer']['accumulation_iters']) * config['trainer']['epochs'])
        params = {
            'optimizer': optimizer,
            'num_training_steps': total_steps,
            'num_warmup_steps': 0.1 * total_steps
        }
        lr_scheduler = getattr(transformers, config['lr_scheduler']['type'])(**params)
    else:
        lr_scheduler = None

    start = time.time()
    print('initializing trainer...')
    Trainer = config.initialize_trainer('arch', trainer_arch)
    trainer = Trainer(model, loss, metrics, pre_metric, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      lr_scheduler=lr_scheduler)
    print('training starts.')
    evaluations = trainer.train()
    end = time.time()
    logger.info(f"Finish training in {end - start} seconds")
    return evaluations


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--suffix', default="", type=str, help='suffix indicating this run (default: None)')
    args.add_argument('-n', '--n_trials', default=1, type=int, help='number of trials (default: 1)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--exp'], type=str, target=('exp_name',)),
        # Data loader (self-supervision generation)
        CustomArgs(['--train_data'], type=str, target=('train_data_loader', 'args', 'data_path')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('train_data_loader', 'args', 'batch_size')),
        CustomArgs(['--ns', '--negative_size'], type=int, target=('train_data_loader', 'args', 'negative_size')),
        CustomArgs(['--ef', '--expand_factor'], type=int, target=('train_data_loader', 'args', 'expand_factor')),
        CustomArgs(['--crt', '--cache_refresh_time'], type=int,
                   target=('train_data_loader', 'args', 'cache_refresh_time')),
        CustomArgs(['--nw', '--num_workers'], type=int, target=('train_data_loader', 'args', 'num_workers')),
        CustomArgs(['--sm', '--sampling_mode'], type=int, target=('train_data_loader', 'args', 'sampling_mode')),
        # Trainer & Optimizer
        CustomArgs(['--mode'], type=str, target=('mode',)),
        CustomArgs(['--loss'], type=str, target=('loss',)),
        CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--es', '--early_stop'], type=int, target=('trainer', 'early_stop')),
        CustomArgs(['--tbs', '--test_batch_size'], type=int, target=('trainer', 'test_batch_size')),
        CustomArgs(['--v', '--verbose_level'], type=int, target=('trainer', 'verbosity')),
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--wd', '--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay')),
        CustomArgs(['--l1'], type=float, target=('trainer', 'l1')),
        CustomArgs(['--l2'], type=float, target=('trainer', 'l2')),
        CustomArgs(['--l3'], type=float, target=('trainer', 'l3')),
        # Model architecture
        CustomArgs(['--pm', '--propagation_method'], type=str, target=('arch', 'args', 'propagation_method')),
        CustomArgs(['--rm', '--readout_method'], type=str, target=('arch', 'args', 'readout_method')),
        CustomArgs(['--mm', '--matching_method'], type=str, target=('arch', 'args', 'matching_method')),
        CustomArgs(['--k'], type=int, target=('arch', 'args', 'k')),
        CustomArgs(['--in_dim'], type=int, target=('arch', 'args', 'in_dim')),
        CustomArgs(['--hidden_dim'], type=int, target=('arch', 'args', 'hidden_dim')),
        CustomArgs(['--out_dim'], type=int, target=('arch', 'args', 'out_dim')),
        CustomArgs(['--pos_dim'], type=int, target=('arch', 'args', 'pos_dim')),
        CustomArgs(['--num_heads'], type=int, target=('arch', 'args', 'heads', 0)),
        CustomArgs(['--feat_drop'], type=float, target=('arch', 'args', 'feat_drop')),
        CustomArgs(['--attn_drop'], type=float, target=('arch', 'args', 'attn_drop')),
        CustomArgs(['--hidden_drop'], type=float, target=('arch', 'args', 'hidden_drop')),
        CustomArgs(['--out_drop'], type=float, target=('arch', 'args', 'out_drop')),
        # TC added
        CustomArgs(['--lp'], type=float, target=('trainer', 'lp')),
        CustomArgs(['--lc'], type=float, target=('trainer', 'lc')),
        CustomArgs(['--lb'], type=float, target=('trainer', 'lb')),
        CustomArgs(['--lw'], type=float, target=('trainer', 'lw')),
    ]
    config = ConfigParser(args, options)
    args = args.parse_args()
    n_trials = args.n_trials

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    setup_seed(47)

    if n_trials > 0:
        config.get_logger('train').info(f'number of trials: {n_trials}')
        metrics = config['metrics']
        save_file = config.log_dir / 'evaluations.txt'
        fin = open(save_file, 'w')
        fin.write('\t'.join(metrics))

        evaluations = []
        exp_name = config['exp_name']

        for i in range(n_trials):
            config.config['exp_name'] = exp_name + '_main'
            config.set_save_dir(i + 1)
            res = main(config)
            evaluations.append(res)
            fin.write('\t'.join([f'{i:.3f}' for i in res]))
            wandb.finish()

        evaluations = np.array(evaluations)
        means = evaluations.mean(axis=0)
        stds = evaluations.std(axis=0)
        final_output = '  '.join([f'& {i:.3f} +- {j:.3f}' for i, j in zip(means, stds)])
        fin.write(final_output)
        config.get_logger('train').info(final_output)
    else:
        main(config)
