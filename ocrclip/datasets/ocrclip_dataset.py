# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import copy
import tempfile
import os.path as osp
import bisect
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.builder import DATASETS
from mmdet.datasets import ConcatDataset, build_dataset
from mmcv.utils import print_log

from mmocr.datasets.icdar_dataset import IcdarDataset
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.datasets import UniformConcatDataset
import mmocr.utils as utils
from mmocr.utils import get_root_logger, is_2dlist, is_type_list

from mmocr.core.evaluation.hmean import eval_hmean
from num2words import num2words

'''icdar(coco) style dataset'''
@DATASETS.register_module()
class OCRCLIPDataset(IcdarDataset):
    CLASSES = ('text',)
    # PROMPT_CLASSES = ('a set of many arbitrary-shape text instances.',)
    # PROMPT_CLASSES = ('the set of many arbitrary-shape text instances.',)
    PROMPT_CLASSES = ('the pixels of many arbitrary-shape text instances.',)
    # PROMPT_CLASSES = ('a set of texts, including horizontal, multi-oriented and curved text.',)
    # INSTANCE_CLASSES = ('first text', 'second text')
    INSTANCE_CLASSES = [num2words(x, ordinal=True) + ' text' for x in range(1, 101)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_name='default_name',
                 postprocessor_cfg=None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 select_first_k=-1):
        # select first k images for fast debugging.
        # self.select_first_k = select_first_k
        self.data_name = data_name
        self.postprocessor_cfg = postprocessor_cfg
        IcdarDataset.__init__(self, ann_file, pipeline, classes, data_root, img_prefix,
                              seg_prefix, proposal_file, test_mode, filter_empty_gt, select_first_k)

        self.bad_img_file_name = []

    def _log_filter_error_index(self, err_index, replace_idx):
        """Logging data info of bad index."""
        try:
            data_info = self.data_infos[err_index]
            img_prefix = self.img_prefix
            print_log(f'Warning: skip broken error file {data_info} '
                      f'with img_prefix {img_prefix}, '
                      f'replace with index {replace_idx}, info: {self.data_infos[replace_idx]}')
        except Exception as e:
            print_log(f'load filter index {err_index} with error {e}')

    def filter_unhandled_img_in_synthtext(self, index):

        if self.data_name != 'syntext150k': return index

        data_info = self.data_infos[index]
        file_name = data_info['file_name']

        # idx = ['82592', '82567', '82629', '82652', '343383']
        if file_name not in self.bad_img_file_name: return index

        new_index = (index + 1) % len(self)
        # self._log_filter_error_index(index, new_index)
        return new_index


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        try:
            # idx = self.filter_unhandled_img_in_synthtext(idx)
            img_info = self.data_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results)
            return self.pipeline(results)
        except Exception as e:
            data_info = self.data_infos[idx]
            img_prefix = self.img_prefix
            print_log(f'Warning: skip broken error file {data_info} '
                      f'with img_prefix {img_prefix}, ')
            idx = (idx + 1) % len(self)
            return self.prepare_train_img(idx)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []

        count = 0
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            count = count + 1
            if count >= self.select_first_k and self.select_first_k > 0:
                break
        print_log(f'Loaded {self.data_name} {len(data_infos)} images', logger=get_root_logger())
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore, seg_map. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ignore = []
        gt_masks_ann = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
                gt_masks_ignore.append(ann.get(
                    'segmentation', None))  # to float32 for latter processing

            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 logger=None,
                 score_thr=0.3,
                 rank_list=None,
                 **kwargs):
        """Evaluate the hmean metric.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[dict[str: float]]: The evaluation results.
        """
        assert utils.is_type_list(results, dict)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-iou', 'hmean-ic13']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_info = {'filename': self.data_infos[i]['file_name']}
            img_infos.append(img_info)
            ann_infos.append(self.get_ann_info(i))

        eval_results = eval_hmean(
            results,
            img_infos,
            ann_infos,
            metrics=metrics,
            score_thr=score_thr,
            logger=logger,
            rank_list=rank_list)
        tmp_res = {}
        for k, v in eval_results.items():
            tmp_res[f'{self.data_name}_{k}'] = v
        return tmp_res

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir




'''icdar(coco) style dataset'''
@DATASETS.register_module()
class VisOCRCLIPDataset(OCRCLIPDataset):
    CLASSES = ('text',)
    # PROMPT_CLASSES = ('a set of many arbitrary-shape text instances.',)
    # PROMPT_CLASSES = ('the set of many arbitrary-shape text instances.',)
    PROMPT_CLASSES = ('the pixels of many arbitrary-shape text instances.',)
    # PROMPT_CLASSES = ('a set of texts, including horizontal, multi-oriented and curved text.',)
    # INSTANCE_CLASSES = ('first text', 'second text')
    INSTANCE_CLASSES = [num2words(x, ordinal=True) + ' text' for x in range(1, 101)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_name='default_name',
                 postprocessor_cfg=None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 select_first_k=-1):
        # select first k images for fast debugging.
        # self.select_first_k = select_first_k
        self.data_name = data_name
        self.postprocessor_cfg = postprocessor_cfg
        IcdarDataset.__init__(self, ann_file, pipeline, classes, data_root, img_prefix,
                              seg_prefix, proposal_file, test_mode, filter_empty_gt, select_first_k)

        self.bad_img_file_name = []

    def _log_filter_error_index(self, err_index, replace_idx):
        """Logging data info of bad index."""
        try:
            data_info = self.data_infos[err_index]
            img_prefix = self.img_prefix
            print_log(f'Warning: skip broken error file {data_info} '
                      f'with img_prefix {img_prefix}, '
                      f'replace with index {replace_idx}, info: {self.data_infos[replace_idx]}')
        except Exception as e:
            print_log(f'load filter index {err_index} with error {e}')

    def filter_unhandled_img_in_synthtext(self, index):

        if self.data_name != 'syntext150k': return index

        data_info = self.data_infos[index]
        file_name = data_info['file_name']

        # idx = ['82592', '82567', '82629', '82652', '343383']
        if file_name not in self.bad_img_file_name: return index

        new_index = (index + 1) % len(self)
        # self._log_filter_error_index(index, new_index)
        return new_index

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        try:
            # idx = self.filter_unhandled_img_in_synthtext(idx)
            img_info = self.data_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results)
            return self.pipeline(results)
        except Exception as e:
            data_info = self.data_infos[idx]
            img_prefix = self.img_prefix
            print_log(f'Warning: skip broken error file {data_info} '
                      f'with img_prefix {img_prefix}, ')
            idx = (idx + 1) % len(self)
            return self.prepare_train_img(idx)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        # ic15
        # keep_imgs = ['img_48.jpg', 'img_68.jpg']
        keep_imgs = ['img_48.jpg', 'img_68.jpg','img_72.jpg','img_75.jpg', 'img_79.jpg','img_82.jpg','img_88.jpg','img_108.jpg'
                    'img_112.jpg','img_114.jpg','img_121.jpg','img_142.jpg','img_143.jpg','img_152.jpg'
                     ]

        count = 0
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            tmp_name = osp.basename(info['file_name'] )
            # if tmp_name not in keep_imgs:
            #     continue
            info['filename'] = info['file_name']
            data_infos.append(info)
            count = count + 1
            if count >= self.select_first_k and self.select_first_k > 0:
                break
        print_log(f'Loaded {self.data_name} {len(data_infos)} images', logger=get_root_logger())
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore, seg_map. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ignore = []
        gt_masks_ann = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
                gt_masks_ignore.append(ann.get(
                    'segmentation', None))  # to float32 for latter processing

            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 logger=None,
                 score_thr=0.3,
                 rank_list=None,
                 **kwargs):
        """Evaluate the hmean metric.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[dict[str: float]]: The evaluation results.
        """
        assert utils.is_type_list(results, dict)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-iou', 'hmean-ic13']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_info = {'filename': self.data_infos[i]['file_name']}
            img_infos.append(img_info)
            ann_infos.append(self.get_ann_info(i))

        eval_results = eval_hmean(
            results,
            img_infos,
            ann_infos,
            metrics=metrics,
            score_thr=score_thr,
            logger=logger,
            rank_list=rank_list)
        tmp_res = {}
        for k, v in eval_results.items():
            tmp_res[f'{self.data_name}_{k}'] = v
        return tmp_res

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir



'''normal det dataset, LmdbLoader or HardDiskLoader, LineJsonParser or LineStrParser'''


@DATASETS.register_module()
class OCRCLIPDetDataset(BaseDataset):

    def __init__(self, data_name='default_name', postprocessor_cfg=None, **kwargs):
        super(OCRCLIPDetDataset, self).__init__(**kwargs)
        self.data_name = data_name
        self.postprocessor_cfg = postprocessor_cfg
        print_log(f'Loaded {self.data_name} {len(self)} images', logger=get_root_logger())
        # temp solution
        self.bad_img_file_name = ['67/fruits_129_40.jpg', '67/fruits_129_18.jpg', '67/fruits_129_74.jpg',
                                  '67/fruits_129_95.jpg', '194/window_19_96.jpg']

    def _parse_anno_info(self, annotations):
        """Parse bbox and mask annotation.
        Args:
            annotations (dict): Annotations of one image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes, gt_bboxes_ignore = [], []
        gt_masks, gt_masks_ignore = [], []
        gt_labels = []
        for ann in annotations:
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(ann['bbox'])
                gt_masks_ignore.append(ann.get('segmentation', None))
            else:
                gt_bboxes.append(ann['bbox'])
                gt_labels.append(ann['category_id'])
                gt_masks.append(ann.get('segmentation', None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks)

        return ann

    def _log_filter_error_index(self, err_index, replace_idx):
        """Logging data info of bad index."""
        try:
            data_info = self.data_infos[err_index]
            img_prefix = self.img_prefix
            print_log(f'Warning: skip broken error file {data_info} '
                      f'with img_prefix {img_prefix}, '
                      f'replace with index {replace_idx}, info: {self.data_infos[replace_idx]}')
        except Exception as e:
            print_log(f'load filter index {err_index} with error {e}')

    def filter_unhandled_img_in_synthtext(self, index):

        if self.data_name != 'synthtext': return index

        data_info = self.data_infos[index]
        file_name = data_info['file_name']

        # idx = ['82592', '82567', '82629', '82652', '343383']
        if file_name not in self.bad_img_file_name: return index

        new_index = (index + 1) % len(self)
        # self._log_filter_error_index(index, new_index)
        return new_index

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        index = self.filter_unhandled_img_in_synthtext(index)
        img_ann_info = self.data_infos[index]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width']
        }
        ann_info = self._parse_anno_info(img_ann_info['annotations'])
        results = dict(img_info=img_info, ann_info=ann_info)
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        self.pre_pipeline(results)

        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 score_thr=0.3,
                 rank_list=None,
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            score_thr (float): Score threshold for prediction map.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[str: float]
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-iou', 'hmean-ic13']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_ann_info = self.data_infos[i]
            img_info = {'filename': img_ann_info['file_name']}
            ann_info = self._parse_anno_info(img_ann_info['annotations'])
            img_infos.append(img_info)
            ann_infos.append(ann_info)

        eval_results = eval_hmean(
            results,
            img_infos,
            ann_infos,
            metrics=metrics,
            score_thr=score_thr,
            logger=logger,
            rank_list=rank_list)
        tmp_res = {}
        for k, v in eval_results.items():
            tmp_res[f'{self.data_name}_{k}'] = v
        return tmp_res


@DATASETS.register_module()
class PostcfgUniformConcatDataset(UniformConcatDataset):
    """A wrapper of UniformConcatDataset which support get postprocessor cfg
        used for dynamic changing postprocessor of the model when single_gpu_test or multi_gou_test.
        Args:
            separate_postprocessor: if true, replace postprocessor of model in single_gpu_test or multi_gou_test
    """
    def __init__(self, separate_postprocessor=False, **kwargs):
        super(UniformConcatDataset, self).__init__(**kwargs)
        self.separate_postprocessor=separate_postprocessor


    def get_postprocessor_cfg(self, idx):
        """Get postprocess cfg of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Postprocessor cfg
                  used for dynamic changing postprocessor of the model when single_gpu_test or multi_gou_test.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')

        if not all([isinstance(ds, OCRCLIPDataset) or isinstance(ds, OCRCLIPDetDataset) for ds in self.datasets]):
            raise NotImplementedError(
                'Get postprocessor cfg is only'
                ' supported by OCRCLIPDataset or OCRCLIPDetDataset now.')
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        return self.datasets[dataset_idx].postprocessor_cfg


