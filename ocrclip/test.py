#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test
from mmdet.core import encode_mask_results

from mmocr.apis.utils import (disable_text_recog_aug_test,
                              replace_image_to_tensor)
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector
from mmocr.utils import revert_sync_batchnorm

import ocrclip
import datasets
import hooks

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMOCR test (and eval) a model.')
    parser.add_argument('config', help='Test config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('--out', help='Output result file in pickle format.')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed.')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without performing evaluation. It is'
        'useful when you want to format the results to a specific format and '
        'submit them to the test server.')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='The evaluation metrics, which depends on the dataset, e.g.,'
        '"bbox", "seg", "proposal" for COCO, and "mAP", "recall" for'
        'PASCAL VOC. hmean-iou for det_ocr')
    parser.add_argument('--show', action='store_true', help='Show results.')
    parser.add_argument(
        '--show-with-score-map',
        action='store_true',
        help='Format the output results without performing evaluation. It is'
             'used for showing combined score map, pred_text_mask, pred and gt')
    parser.add_argument(
        '--show-dir', help='Directory where the output images will be saved.')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='Score threshold (default: 0.3).')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='Whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='The tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into the config file. If the value '
        'to be overwritten is a list, it should be of the form of either '
        'key="[a,b]" or key=a,b. The argument also allows nested list/tuple '
        'values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks '
        'are necessary and that no white space is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='Custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='Custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Options for job launcher.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options.')
        args.eval_options = args.options
    return args


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    is_kie=False,
                    show_score_thr=0.3,
                    show_with_score_map=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # torch.cuda.empty_cache()
        with torch.no_grad():
            if 'ann_info' in data:
                gt_masks = data.pop('ann_info', None) # [DataContainer] for visualization
            result = model(return_loss=False, rescale=True, **data)

        # import pdb; pdb.set_trace()
        batch_size = len(result)
        if show or out_dir:
            if is_kie:
                img_tensor = data['img'].data[0]
                if img_tensor.shape[0] != 1:
                    raise KeyError('Visualizing KIE outputs in batches is'
                                   'currently not supported.')
                gt_bboxes = data['gt_bboxes'].data[0]
                img_metas = data['img_metas'].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                for i, img in enumerate(imgs):
                    h, w, _ = img_metas[i]['img_shape']
                    img_show = img[:h, :w, :]
                    if out_dir:
                        out_file = osp.join(out_dir,
                                            img_metas[i]['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        gt_bboxes[i],
                        show=show,
                        out_file=out_file)
            else:
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                # import pdb; pdb.set_trace()
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)
                # import pdb; pdb.set_trace()
                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :] # remove padding

                    ori_h, ori_w = img_meta['ori_shape'][:-1] # recover to ori_sz
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None
                    if show_with_score_map:
                        if gt_masks is not None:
                            gt_masks = gt_masks[0].data[0][0]
                            flat_gt_masks = []
                            for mask_instance in gt_masks['masks']:
                                flat_gt_masks.append(mask_instance[0])
                            # print(flat_gt_masks)
                        else:
                            flat_gt_masks=None
                        model.module.show_result_with_score_map(
                            img_show,
                            result[i],
                            gt_masks=flat_gt_masks,
                            img_meta=img_meta,
                            show=show,
                            out_file=out_file,
                            score_thr=show_score_thr)
                    else:
                        model.module.show_result(
                            img_show,
                            result[i],
                            show=show,
                            out_file=out_file,
                            score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():
    args = parse_args()

    assert (
        args.out or args.eval or args.format_only or args.show
        or args.show_dir), (
            'Please specify at least one operation (save/eval/format/show the '
            'results / save the results) with the argument "--out", "--eval"'
            ', "--format-only", "--show" or "--show-dir".')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified.')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.model.get('pretrained'):
        cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # 设置backbone的pretrain为None

    # in case the test dataset is concatenated
    samples_per_gpu = (cfg.data.get('test_dataloader', {})).get(
        'samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
    if samples_per_gpu > 1:
        cfg = disable_text_recog_aug_test(cfg)
        cfg = replace_image_to_tensor(cfg)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=False if args.show_with_score_map else True))# load annotation
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'workers_per_gpu',
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get('test_dataloader', {}),
        **dict(samples_per_gpu=samples_per_gpu)
    }

    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if 'OCRCLIP' in cfg.model.type and not hasattr(cfg.model, 'class_names'):
        cfg.model.class_names = list(dataset.CLASSES)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = revert_sync_batchnorm(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    rank, _ = get_dist_info()
    print(f'Config:\n{cfg.pretty_text}') if rank == 0 else None

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        is_kie = cfg.model.type in ['SDMGR']
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  is_kie, args.show_score_thr, show_with_score_map=args.show_with_score_map)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    print('test done') if rank == 0 else None

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            eval_res = dataset.evaluate(outputs, **eval_kwargs)
            round_eval_res = {}
            for eval_key in eval_res.keys():
                eval_v = eval_res[eval_key]
                round_eval_res[eval_key] = round(eval_v, 3)
            print(round_eval_res)


if __name__ == '__main__':
    main()
