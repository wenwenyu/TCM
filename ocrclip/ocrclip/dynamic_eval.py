# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 3/1/22 9:23 PM

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import torch
import torch.distributed as dist

import mmcv
from mmcv.parallel import is_module_wrapper
from mmcv.runner import get_dist_info
from mmcv.engine.test import collect_results_cpu, collect_results_gpu
from mmocr.models.builder import build_postprocessor

def dynamic_single_gpu_test(model, data_loader):
    """
    Change postprocessor of the model depend on dataset cfg
    Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    # remember original postprocessor of model

    model.eval()
    results = []
    dataset = data_loader.dataset

    # new added begin###
    # save original postprocessor of the model
    if dataset.__class__.__name__ == 'PostcfgUniformConcatDataset' and dataset.separate_postprocessor:
        assert data_loader.batch_size==1, \
            "PostcfgUniformConcatDataset setting separate_postprocessor=true only support batch_size=1"
        if is_module_wrapper(model):
            older_postprocessor = model.module.bbox_head.postprocessor
        else:
            older_postprocessor = model.bbox_head.postprocessor
    # new added end###

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # new added begin###
        model = replace_postprocesor(model, dataset, i) # new added
        # new added end###
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    # new added begin###
    # rollback to original postprocessor
    if dataset.__class__.__name__ == 'PostcfgUniformConcatDataset' and dataset.separate_postprocessor:
        # rollback to original postprocessor
        if is_module_wrapper(model):
            model.module.bbox_head.postprocessor = older_postprocessor
        else:
            model.bbox_head.postprocessor = older_postprocessor
    # new added end###

    return results


def dynamic_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """
    Change postprocessor of the model depend on dataset cfg

    Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset

    # new added begin###
    # save original postprocessor of the model
    if dataset.__class__.__name__ == 'PostcfgUniformConcatDataset' and dataset.separate_postprocessor:
        assert data_loader.batch_size==1, \
            "PostcfgUniformConcatDataset setting separate_postprocessor=true only support batch_size=1"
        if is_module_wrapper(model):
            older_postprocessor = model.module.bbox_head.postprocessor
        else:
            older_postprocessor = model.bbox_head.postprocessor
    # new added end###

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        # new added begin###
        model = replace_postprocesor(model, dataset, i) # new added
        # new added end###
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar.completed
            for _ in range(batch_size_all):
                prog_bar.update()

    # new added begin###
    # rollback to original postprocessor
    if dataset.__class__.__name__ == 'PostcfgUniformConcatDataset' and dataset.separate_postprocessor:
        if is_module_wrapper(model):
            model.module.bbox_head.postprocessor = older_postprocessor
        else:
            model.bbox_head.postprocessor = older_postprocessor
    # new added end###

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results

def replace_postprocesor(model, dataset, data_index):
    if dataset.__class__.__name__ == 'PostcfgUniformConcatDataset' and dataset.separate_postprocessor:
        postprocessor_cfg = dataset.get_postprocessor_cfg(data_index)
        if postprocessor_cfg is None:
            return model
        postprocessor = build_postprocessor(postprocessor_cfg)
        if is_module_wrapper(model):
            model.module.bbox_head.postprocessor = postprocessor
        else:
            model.bbox_head.postprocessor = postprocessor
    return model
