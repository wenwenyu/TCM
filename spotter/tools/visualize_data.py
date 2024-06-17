#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import numpy as np
import os
from itertools import chain
import cv2
import tqdm
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from adet.config import get_cfg
from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.projects.testr_clip import add_clip_config
from adet.data.builtin import register_all_coco
from adet.utils.misc import is_2dlist

import sys
# from shapely.geometry import Polygon as plg
from Polygon import Polygon as plg
import pyclipper
from detectron2.data import detection_utils as utils
from mmdet.core import PolygonMasks


def setup(args):
    cfg = get_cfg()
    add_clip_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


def process_polygons(polygons):
    """Convert polygons to list of ndarray and filter invalid polygons.

    Args:
        polygons (list[list]): Polygons of one instance.

    Returns:
        list[numpy.ndarray]: Processed polygons.
    """

    polygons = [np.array(p).astype(np.float32) for p in polygons]
    valid_polygons = []
    for polygon in polygons:
        if len(polygon) % 2 == 0 and len(polygon) >= 6:
            valid_polygons.append(polygon)
    return valid_polygons

def _load_masks(results):
    w, h = results["width"], results["height"]
    # 最里面 `list[float]` 代表 object 一部分 mask 的值
    # 中间 `list[list[float]]` 是一个 object 的 mask, 一个 object 可有多个 mask
    # 最外面 `list[list[float]]` 是一个 image 的 mask, 一个 image 可有多个 mask
    # 'gt_masks': list[list[list[float]]]
    gt_masks = [[instance['sem_seg']] for instance in results['annotations']]

    gt_masks = PolygonMasks(
        [process_polygons(polygons) for polygons in gt_masks], h,
        w)
    gt_shrink, ignore_tag = generate_kernels((h, w), gt_masks.masks, shrink_ratio=0.9)
    return gt_shrink

def generate_kernels(
                     img_size,
                     text_polys,
                     shrink_ratio=0.9,
                     max_shrink=sys.maxsize,
                     ignore_tags=None):
    """Generate text instance kernels for one shrink ratio.

    Args:
        img_size (tuple(int, int)): The image size of (height, width).
        text_polys (list[list[ndarray]]: The list of text polygons.
        shrink_ratio (float): The shrink ratio of kernel.

    Returns:
        text_kernel (ndarray): The text kernel mask of (height, width).
    """
    assert isinstance(img_size, tuple)
    assert is_2dlist(text_polys)
    assert isinstance(shrink_ratio, float)

    h, w = img_size
    # text_kernel = np.zeros((h, w), dtype=np.float32)
    # text_kernel = np.zeros((h, w), dtype=np.int32) # n_class is background
    text_kernel = np.ones((h, w), dtype=np.int32) # n_class is background
    # import pdb;pdb.set_trace()
    for text_ind, poly in enumerate(text_polys):

        instance = poly[0].reshape(-1, 2).astype(np.int32)

        area = plg(instance).area()
        peri = cv2.arcLength(instance, True)
        distance = min(
            int(area * (1 - shrink_ratio * shrink_ratio) / (peri + 0.001) +
                0.5), max_shrink)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(instance, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
        shrinked = np.array(pco.Execute(-distance))

        # check shrinked == [] or empty ndarray
        if len(shrinked) == 0 or shrinked.size == 0:
            if ignore_tags is not None:
                ignore_tags[text_ind] = True
            continue
        try:
            shrinked = np.array(shrinked[0]).reshape(-1, 2)

        except Exception as e:
            print(f'{shrinked} with error {e}')
            if ignore_tags is not None:
                ignore_tags[text_ind] = True
            continue
        # cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)],
        #              text_ind + 1)
        cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)],
                      0)
        # cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)],
        #              1.0)
    return text_kernel, ignore_tags


# python tools/visualize_data.py --source dataloader --config-file configs/TESTR/ICDAR15/TESTR_R_50_Polygon_debug.yaml --output-dir ./output/ic15_da

# python tools/visualize_data.py --source annotation --config-file configs/TESTR/ICDAR15/TESTR_R_50_Polygon_debug.yaml --output-dir ./output/ic15_sem
# python tools/visualize_data.py --source annotation --config-file configs/TESTR/CTW1500/TESTR_R_50_Polygon_debug.yaml --output-dir ./output/ctw_sem
# python tools/visualize_data.py --source annotation --config-file configs/TESTR/TotalText/TESTR_R_50_Polygon_debug.yaml --output-dir ./output/tt_sem
# python tools/visualize_data.py --source annotation --config-file configs/BAText/ICDAR2015/v1_attn_R_50_debug.yaml --output-dir ./output/ic15_bezai_sem
# python tools/visualize_data.py --source annotation --config-file configs/BAText/CTW1500/v1_attn_R_50_debug.yaml --output-dir ./output/ctw_bezai_sem2


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)
    register_all_coco(root=cfg.DATA_ROOT)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 2.0 if args.show else 1.0
    if args.source == "dataloader":
        mapper = DatasetMapperWithBasis(cfg, True)
        train_data_loader = build_detection_train_loader(cfg, mapper)
        for batch in train_data_loader:
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                img = per_image["image"].permute(1, 2, 0)
                if cfg.INPUT.FORMAT == "BGR":
                    img = img[:, :, [2, 1, 0]]
                else:
                    # img = img.numpy()
                    img = np.asarray(Image.fromarray(img.numpy(), mode=cfg.INPUT.FORMAT).convert("RGB"))

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                output(vis, str(per_image["image_id"]) + ".jpg")
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        if cfg.MODEL.KEYPOINT_ON:
            dicts = filter_images_with_few_keypoints(dicts, 1)
        for dic in tqdm.tqdm(dicts):
            if len(dic['annotations']) > 0:
                if 'sem_seg' in dic['annotations'][0]:
                    # overwrite sem_seg_gt for clip score matching label
                    sem_seg_gt = _load_masks(dic)
                    dic['sem_seg'] = sem_seg_gt
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            # import pdb;pdb.set_trace()
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))