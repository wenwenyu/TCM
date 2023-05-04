import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmocr.models import build_detector
import ocrclip
import datasets
import mmdet.ResNet
from fvcore.nn import FlopCountAnalysis

import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from fvcore.nn import FlopCountAnalysis
from mmocr.datasets import build_dataset


def calc_flops(model, img_size):
    with torch.no_grad():
        x = torch.randn(1, img_size[0], img_size[1], img_size[2]).cuda()
        fca1 = FlopCountAnalysis(model, x)
        print('backbone:', fca1.total(module_name="backbone")/1e9)
        try:
            print('text_encoder:', fca1.total(module_name="text_encoder")/1e9)
            print('context_decoder:', fca1.total(module_name="context_decoder")/1e9)
            print('prompt_generator:', fca1.total(module_name="prompt_generator")/1e9)
            print('identity_head:', fca1.total(module_name="identity_head")/1e9)
        except:
            pass
        
        try:
            print('neck:', fca1.total(module_name="neck")/1e9)
        except:
            pass
        print('bbox_head:', fca1.total(module_name="bbox_head")/1e9)
        flops1 = fca1.total()
        print("#### GFLOPs: {:.1f}".format(flops1 / 1e9))
    return flops1 / 1e9

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--fvcore',
        action='store_true', default=False)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        # default=[1024, 1024],
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--size-divisor',
        type=int,
        default=32,
        help='Pad the input image, the minimum size that is divisible '
             'by size_divisor, -1 means do not pad the image.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # if len(args.shape) == 1:
    #     input_shape = (3, args.shape[0], args.shape[0])
    # elif len(args.shape) == 2:
    #     input_shape = (3, ) + tuple(args.shape)
    # else:
    #     raise ValueError('invalid input shape')
    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    orig_shape = (3, h, w)

    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor

    input_shape = (3, h, w)

    if divisor > 0 and \
            input_shape != orig_shape:
        split_line = '=' * 30
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {orig_shape} to {input_shape}\n')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    if 'OCRCLIP' in cfg.model.type:
        datasets = [build_dataset(cfg.data.train)]
        cfg.model.class_names = list(datasets[0].CLASSES)
    
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
        
    if args.fvcore:
        flops = calc_flops(model, input_shape)
        
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print('number of params:', f'{n_parameters:.1f}')
        if hasattr(model, 'text_encoder'):
            n_parameters_text = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad) / 1e6
            print('param without text encoder:', n_parameters-n_parameters_text)
        if hasattr(model, 'context_decoder'):
            n_parameters_text = sum(p.numel() for p in model.context_decoder.parameters() if p.requires_grad) / 1e6
            print('param context:', n_parameters_text)
        if hasattr(model, 'prompt_generator'):
            n_parameters_text = sum(p.numel() for p in model.prompt_generator.parameters() if p.requires_grad) / 1e6
            print('param prompt_generator:', n_parameters_text)

    else:
        flops, params = get_model_complexity_info(model, input_shape)
        split_line = '=' * 30
        print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
            split_line, input_shape, flops, params))
        print('!!!Please be cautious if you use the results in papers. '
              'You may need to check if all ops are supported and verify that the '
              'flops computation is correct.')


if __name__ == '__main__':
    main()

# python get_flops.py configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_td_ranger_post_taiji_1033_vis_feat.py --fvcore
# python get_flops.py configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py --fvcore
# python get_flops.py configs/textdet/dbnet/dbnet_r101dcnv2_fpnc_1200e_icdar2015.py --fvcore
# python get_flops.py configs/textdet/dbnet/dbnet_r152dcnv2_fpnc_1200e_icdar2015.py --fvcore


# python get_flops.py configs/textdet/panet/panet_r50_fpem_ffm_600e_icdar2017.py --fvcore
# python get_flops.py configs/textdet/panet/panet_clip_r50att_prompt_gen_vis_fpem_ffm_600e_ic15_1033_debug.py --fvcore


# python get_flops.py configs/textdet/fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py --fvcore
# python get_flops.py configs/textdet/fcenet/fcenet_clip_r50att_prompt_gen_vis_fpn_1500e_ic15_1033_debug.py --fvcore


# python get_flops.py configs/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py --fvcore

# python get_flops.py configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py --fvcore

# python get_flops.py configs/textdet/drrg/drrg_r50_fpn_unet_1200e_ctw1500.py --fvcore