import cv2
import numpy as np

import torch.nn.functional as F

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from mmocr.models.textdet.postprocess import BasePostprocessor
from mmocr.models.textdet.postprocess.utils import box_score_fast, unclip, fill_hole, fourier2poly, poly_nms

import torch
import pyclipper
from shapely.geometry import Polygon

@POSTPROCESSOR.register_module()
class DBParamPostprocessor(BasePostprocessor):
    """Decoding predictions of DbNet to instances. This is partially adapted
    from https://github.com/MhLiao/DB.
    Add arcLength_ratio param

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        mask_thr (float): The mask threshold value for binarization.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted.
        unclip_ratio (float): The unclip ratio for text regions dilation.
        max_candidates (int): The maximum candidate number.
    """

    def __init__(self,
                 text_repr_type='poly',
                 mask_thr=0.3,
                 min_text_score=0.3,
                 min_text_width=5,
                 unclip_ratio=1.5,
                 with_logits=False,
                 use_approxPolyDP=True,
                 arcLength_ratio=0.01,
                 max_candidates=3000,
                 **kwargs):
        super().__init__(text_repr_type)
        self.mask_thr = mask_thr
        self.min_text_score = min_text_score
        self.min_text_width = min_text_width
        self.unclip_ratio = unclip_ratio
        self.with_logits = with_logits
        self.use_approxPolyDP = use_approxPolyDP
        self.arcLength_ratio = arcLength_ratio
        self.max_candidates = max_candidates

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The predicted text boundaries.
        """
        assert preds.dim() == 3
        if self.with_logits:
            preds = F.sigmoid(preds)

        prob_map = preds[0, :, :]
        text_mask = prob_map > self.mask_thr

        score_map = prob_map.data.cpu().numpy().astype(np.float32)
        text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

        contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boundaries = []
        for i, poly in enumerate(contours):
            if i > self.max_candidates:
                break
            if self.use_approxPolyDP:
                epsilon = self.arcLength_ratio * cv2.arcLength(poly, True)
                approx = cv2.approxPolyDP(poly, epsilon, True)
            else:
                approx = poly
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = box_score_fast(score_map, points)
            if score < self.min_text_score:
                continue
            poly = unclip(points, unclip_ratio=self.unclip_ratio)
            if len(poly) == 0 or isinstance(poly[0], list):
                continue
            poly = poly.reshape(-1, 2)

            if self.text_repr_type == 'quad':
                poly = points2boundary(poly, self.text_repr_type, score,
                                       self.min_text_width)
            elif self.text_repr_type == 'poly':
                poly = poly.flatten().tolist()
                if score is not None:
                    poly = poly + [score]
                if len(poly) < 8:
                    poly = None

            if poly is not None:
                boundaries.append(poly)

        return boundaries



@POSTPROCESSOR.register_module()
class DBVisPostprocessor(BasePostprocessor):
    """Decoding predictions of DbNet to instances. This is partially adapted
    from https://github.com/MhLiao/DB.
    Add arcLength_ratio param, return_vis_map, filter_min_text_poly

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        mask_thr (float): The mask threshold value for binarization.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted.
        unclip_ratio (float): The unclip ratio for text regions dilation.
        max_candidates (int): The maximum candidate number.
    """

    def __init__(self,
                 text_repr_type='poly',
                 mask_thr=0.3,
                 min_text_score=0.3,
                 min_text_width=5,
                 unclip_ratio=1.5,
                 return_vis_map=False,
                 use_approxPolyDP=True,
                 filter_min_text_poly=False,
                 arcLength_ratio=0.01,
                 max_candidates=3000,
                 **kwargs):
        super().__init__(text_repr_type)
        self.mask_thr = mask_thr
        self.min_text_score = min_text_score
        self.min_text_width = min_text_width
        self.unclip_ratio = unclip_ratio
        self.use_approxPolyDP = use_approxPolyDP
        self.arcLength_ratio = arcLength_ratio
        self.max_candidates = max_candidates
        self.return_vis_map = return_vis_map
        self.filter_min_text_poly = filter_min_text_poly

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The predicted text boundaries., adding intermedia features
        """
        assert preds.dim() == 3

        prob_map = preds[0, :, :]
        text_mask = prob_map > self.mask_thr

        score_map = prob_map.data.cpu().numpy().astype(np.float32)
        text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

        contours, _ = cv2.findContours((text_mask * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # import pdb;pdb.set_trace()
        boundaries = []
        for i, poly in enumerate(contours):
            if i > self.max_candidates:
                break
            if self.use_approxPolyDP:
                epsilon = self.arcLength_ratio * cv2.arcLength(poly, True)
                approx = cv2.approxPolyDP(poly, epsilon, True)
            else:
                approx = poly # (-1, 1, 2)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = box_score_fast(score_map, points)
            if score < self.min_text_score:
                continue
            poly = unclip(points, unclip_ratio=self.unclip_ratio) # (1, -1, 2) # db poly is 2.0
            if len(poly) == 0 or isinstance(poly[0], list):
                continue
            poly = poly.reshape(-1, 2)

            if self.text_repr_type == 'quad':
                poly = points2boundary(poly, self.text_repr_type, score,
                                       self.min_text_width)
            elif self.text_repr_type == 'poly':
                if self.filter_min_text_poly: # db default is True
                    _, sside = self.get_mini_boxes(poly.reshape((-1, 1, 2)))
                    if sside < self.min_text_width: # db is min_text_width + 2
                        continue
                poly = poly.flatten().tolist()
                if score is not None:
                    poly = poly + [score]
                if len(poly) < 8:
                    poly = None

            if poly is not None:
                boundaries.append(poly)
        if self.return_vis_map:
            return (boundaries, score_map, text_mask)
        else:
            return boundaries

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])




@POSTPROCESSOR.register_module()
class TextSegPostprocessor(BasePostprocessor):
    """Decoding predictions of TextSeg to instances. This is partially adapted
    from https://github.com/MhLiao/DB.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        mask_thr (float): The mask threshold value for binarization.
        min_text_score (float): The threshold value for converting binary map
            to shrink text regions.
        min_text_width (int): The minimum width of boundary polygon/box
            predicted.
        unclip_ratio (float): The unclip ratio for text regions dilation.
        max_candidates (int): The maximum candidate number.
    """

    def __init__(self,
                 text_repr_type='poly',
                 mask_thr=0.3,
                 min_text_score=0.6,
                 min_text_width=3,
                 unclip_ratio=1.5,
                 with_logits=True,
                 return_vis_map=False,
                 use_approxPolyDP=True,
                 arcLength_ratio=0.01,
                 max_candidates=1000,
                 **kwargs):
        super().__init__(text_repr_type)
        self.mask_thr = mask_thr
        self.min_text_score = min_text_score
        self.min_text_width = min_text_width
        self.unclip_ratio = unclip_ratio
        self.with_logits=with_logits
        self.use_approxPolyDP = use_approxPolyDP
        self.arcLength_ratio = arcLength_ratio
        self.max_candidates = max_candidates
        self.return_vis_map = return_vis_map

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`, [0,1] score map.

        Returns:
            list[list[float]]: The predicted text boundaries with score.
        """
        assert preds.dim() == 3

        if self.with_logits:
            preds = F.sigmoid(preds)
        prob_map = preds[0, :, :]
        text_mask = prob_map > self.mask_thr

        score_map = prob_map.data.cpu().numpy().astype(np.float32)
        text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

        if self.text_repr_type == 'quad':
            boxes = self.boxes_from_bitmap(score_map, text_mask)
        elif self.text_repr_type == 'poly':
            boxes = self.polygons_from_bitmap(score_map, text_mask)

        #  list of predicted text boundaries with score
        if self.return_vis_map:
            return (boxes, score_map, text_mask)
        else:
            return boxes

    def boxes_from_bitmap(self, pred, _bitmap):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        # height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]
        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_text_width:
                continue
            points = np.array(points)
            score = box_score_fast(pred, points.reshape(-1, 2))
            if self.min_text_score > score:
                continue
            box = unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_text_width + 2:
                continue
            box = np.array(box).reshape(-1, 2)
            box = box.flatten().tolist()
            if score is not None:
                box = box + [score]
            if len(box) < 8: continue
            boxes.append(box)
        return boxes

    def polygons_from_bitmap(self, pred, _bitmap):
        '''
        pred: predicted score map
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        # height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]
        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        for index in range(num_contours):
            contour = contours[index] # (*, 1, 2)
            if self.use_approxPolyDP:
                epsilon = self.arcLength_ratio * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
            else:
                approx = contour
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_text_width:
            #     continue
            score = box_score_fast(pred, points.reshape(-1, 2))
            if self.min_text_score > score:
                continue
            if points.shape[0] > 2:
                # box = unclip(points, unclip_ratio=2.0)
                box = unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_text_width + 2:
                continue
            # box = np.array(box)
            box = box.flatten().tolist()
            if score is not None:
                box = box + [score]
            if len(box) < 8: continue
            boxes.append(box)
        return boxes


    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])



@POSTPROCESSOR.register_module()
class MyFCEPostprocessor(BasePostprocessor):
    """Decoding predictions of FCENet to instances.

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        num_reconstr_points (int): The points number of the polygon
            reconstructed from predicted Fourier coefficients.
        text_repr_type (str): Boundary encoding type 'poly' or 'quad'.
        scale (int): The down-sample scale of the prediction.
        alpha (float): The parameter to calculate final scores. Score_{final}
                = (Score_{text region} ^ alpha)
                * (Score_{text center region}^ beta)
        beta (float): The parameter to calculate final score.
        score_thr (float): The threshold used to filter out the final
            candidates.
        nms_thr (float): The threshold of nms.
    """

    def __init__(self,
                 fourier_degree,
                 num_reconstr_points,
                 text_repr_type='poly',
                 alpha=1.0,
                 beta=2.0,
                 score_thr=0.3,
                 nms_thr=0.1,
                 **kwargs):
        super().__init__(text_repr_type)
        self.fourier_degree = fourier_degree
        self.num_reconstr_points = num_reconstr_points
        self.alpha = alpha
        self.beta = beta
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    def __call__(self, preds, scale):
        """
        Args:
            preds (list[Tensor]): Classification prediction and regression
                prediction.
            scale (float): Scale of current layer.

        Returns:
            list[list[float]]: The instance boundary and confidence.
        """
        assert isinstance(preds, list)
        assert len(preds) == 2

        cls_pred = preds[0][0]
        tr_pred = cls_pred[0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = cls_pred[2:].softmax(dim=0).data.cpu().numpy()

        reg_pred = preds[1][0].permute(1, 2, 0).data.cpu().numpy()
        x_pred = reg_pred[:, :, :2 * self.fourier_degree + 1]
        y_pred = reg_pred[:, :, 2 * self.fourier_degree + 1:]

        score_pred = (tr_pred[1]**self.alpha) * (tcl_pred[1]**self.beta)
        tr_pred_mask = (score_pred) > self.score_thr
        tr_mask = fill_hole(tr_pred_mask)

        tr_contours, _ = cv2.findContours(
            tr_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(tr_mask)
        boundaries = []
        for cont in tr_contours:
            deal_map = mask.copy().astype(np.int8)
            cv2.drawContours(deal_map, [cont], -1, 1, -1)

            score_map = score_pred * deal_map
            score_mask = score_map > 0
            xy_text = np.argwhere(score_mask)
            dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

            x, y = x_pred[score_mask], y_pred[score_mask]
            c = x + y * 1j
            c[:, self.fourier_degree] = c[:, self.fourier_degree] + dxy
            c *= scale

            polygons = fourier2poly(c, self.num_reconstr_points)
            score = score_map[score_mask].reshape(-1, 1)
            polygons = poly_nms(
                np.hstack((polygons, score)).tolist(), self.nms_thr)

            boundaries = boundaries + polygons

        boundaries = poly_nms(boundaries, self.nms_thr)

        if self.text_repr_type == 'quad':
            new_boundaries = []
            for boundary in boundaries:
                poly = np.array(boundary[:-1]).reshape(-1,
                                                       2).astype(np.float32)
                score = boundary[-1]
                points = cv2.boxPoints(cv2.minAreaRect(poly))
                points = np.int0(points)
                new_boundaries.append(points.reshape(-1).tolist() + [score])

        return boundaries

