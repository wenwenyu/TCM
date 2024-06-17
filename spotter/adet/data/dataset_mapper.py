import copy
import logging
import os.path as osp
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import sys
import random

# from shapely.geometry import Polygon as plg
from Polygon import Polygon as plg
import torch
import torch.utils.data as data

from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils
import cv2
import pyclipper

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode
from detectron2.data.common import MapDataset
from detectron2.utils.serialize import PicklableWrapper

from mmdet.core import BitmapMasks, PolygonMasks

from .augmentation import RandomCropWithInstance
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations)

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        # used for cilp matching gt
        if 'sem_seg' in dataset_dict['annotations'][0]:
            # overwrite sem_seg_gt for clip score matching label
            sem_seg_gt = self._load_masks(dataset_dict)
        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict


    def process_polygons(self, polygons):
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

    def _load_masks(self, results):
        w, h = results["width"], results["height"]
        # 最里面 `list[float]` 代表 object 一部分 mask 的值
        # 中间 `list[list[float]]` 是一个 object 的 mask, 一个 object 可有多个 mask
        # 最外面 `list[list[float]]` 是一个 image 的 mask, 一个 image 可有多个 mask
        # 'gt_masks': list[list[list[float]]]
        gt_masks = [[instance['sem_seg']] for instance in results['annotations']]

        gt_masks = PolygonMasks(
            [self.process_polygons(polygons) for polygons in gt_masks], h,
            w)
        gt_shrink, ignore_tag = self.generate_kernels((h, w), gt_masks.masks, shrink_ratio=0.9)
        return gt_shrink

    def generate_kernels(self,
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
        text_kernel = np.zeros((h, w), dtype=np.float32)
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
            # shrinked = np.array(pco.Execute(-distance), dtype=object)

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
            cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)],
                         text_ind + 1)
            # cv2.fillPoly(text_kernel, [shrinked.astype(np.int32)],
            #              1.0)
        return text_kernel, ignore_tags

def is_2dlist(x):
    """check x is 2d-list([[1], []]) or 1d empty list([]).

    Notice:
        The reason that it contains 1d empty list is because
        some arguments from gt annotation file or model prediction
        may be empty, but usually, it should be 2d-list.
    """
    if not isinstance(x, list):
        return False
    if len(x) == 0:
        return True

    return all(isinstance(item, list) for item in x)


class TryMapDataset(MapDataset):
    """
    Map a function over the elements in a dataset.
    """
    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            try:
                data = self._map_func(self._dataset[cur_idx])
            except Exception as e:
                self._fallback_candidates.discard(cur_idx)
                data = None
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


