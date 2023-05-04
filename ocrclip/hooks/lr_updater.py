# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 12/30/21 9:36 PM

import datetime

import torch
from mmcv.runner.hooks import HOOKS, LrUpdaterHook

@HOOKS.register_module()
class ResumeLrUpdaterHook(LrUpdaterHook):

    def __init__(self, **kwargs):
        super(ResumeLrUpdaterHook, self).__init__(**kwargs)

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                # group['initial_lr'] for group in runner.optimizer.param_groups
                group['lr'] for group in runner.optimizer.param_groups # change base_lr to latest lr of resume epoch
            ]

@HOOKS.register_module()
class ResumePolyLrUpdaterHook(ResumeLrUpdaterHook):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(ResumePolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr