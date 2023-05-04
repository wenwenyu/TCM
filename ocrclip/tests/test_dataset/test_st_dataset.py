# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 1/7/22 1:40 PM



import copy
import tempfile
from functools import partial
from os.path import dirname, exists, join

import sys
sys.path.append(dirname(dirname(dirname(dirname(__file__)))))
# sys.path.append(join(dirname(dirname(dirname(__file__))), 'ocrclip'))
sys.path.append(join(dirname(dirname(dirname(__file__))), 'datasets'))
# sys.path.append(join(dirname(dirname(dirname(__file__))), 'hooks'))
# sys.path.append(dirname(dirname(dirname(__file__))))
# sys.path.append(dirname(dirname(__file__)))

import numpy as np
import pytest
import torch



from mmocr.utils import revert_sync_batchnorm
from mmocr.datasets import build_dataset
from mmdet.datasets import build_dataloader
from torch.utils.data import DataLoader
from mmcv.parallel import collate
from tqdm import tqdm

# import ocrclip.ocrclip
# import ocrclip.datasets
# import ocrclip.hooks

# import ocrclip
# import datasets
# import hooks

def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmocr repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmocr
        repo_dpath = dirname(dirname(mmocr.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    config = copy.deepcopy(config)
    return config

def test_synth_datast(cfg_file):
    cfg = _get_cfg(cfg_file)
    train_dataset = build_dataset(cfg.data.train)

    # distributed = False
    # # prepare data loaders
    # # step 1: give default values and override (if exist) from cfg.data
    # loader_cfg = {
    #     **dict(
    #         seed=cfg.get('seed'),
    #         drop_last=False,
    #         dist=distributed,
    #         num_gpus=len(cfg.gpu_ids)),
    #     **({} if torch.__version__ != 'parrots' else dict(
    #         prefetch_num=2,
    #         pin_memory=False,
    #     )),
    #     **dict((k, cfg.data[k]) for k in [
    #         'samples_per_gpu',
    #         'workers_per_gpu',
    #         'shuffle',
    #         'seed',
    #         'drop_last',
    #         'prefetch_num',
    #         'pin_memory',
    #         'persistent_workers',
    #     ] if k in cfg.data)
    # }
    #
    # step 2: cfg.data.train_dataloader has highest priority
    # train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))
    #
    # data_loaders = build_dataloader(train_dataset, **train_loader_cfg)

    bs = 64
    data_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        sampler=None,
        num_workers=16,
        batch_sampler=None,
        collate_fn=partial(collate, samples_per_gpu=bs),
        pin_memory=False,
        worker_init_fn=None,
        )
    print(f'total step: {len(data_loader)}')
    try:
        for idx, data in enumerate(data_loader):
            print(idx)
    except Exception as e:
        print(f'error_img: {e.args}')


if __name__ == '__main__':
    cfg_file  = 'textdet/dbnet/clip_dbnet_r50_fpnc_2e_st_real3_pretrain.py'
    # cfg_file  = 'textdet/dbnet/clip_dbnet_r50_fpnc_2e_st_real3_pretrain_taiji.py'

    # test_synth_datast(cfg_file)

    cfg = _get_cfg(cfg_file)
    if hasattr(cfg.model, 'class_names'):
        print('111')
    else:
        print('222')