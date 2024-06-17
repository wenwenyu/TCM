from typing import List, Optional
import torch
from torch.functional import Tensor
from torchvision.ops.boxes import box_area
import torch.distributed as dist
import numpy as np
import functools

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    if target.ndim == 2:
        assert output.ndim == 3
        output = output.mean(1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, -1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def sigmoid_offset(x, offset=True):
    # modified sigmoid for range [-0.5, 1.5]
    if offset:
        return x.sigmoid() * 2 - 0.5
    else:
        return x.sigmoid()

def inverse_sigmoid_offset(x, eps=1e-5, offset=True):
    if offset:
        x = (x + 0.5) / 2.0
    return inverse_sigmoid(x, eps)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


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


def is_type_list(x, type):

    if not isinstance(x, list):
        return False

    return all(isinstance(item, type) for item in x)

def bezier_to_polygon(bezier_points, num_sample=20):
    """Sample points from the boundary of a polygon enclosed by two Bezier
    curves, which are controlled by ``bezier_points``.
    Args:
        bezier_points (ndarray): A :math:`(2, 4, 2)` array of 8 Bezeir points
            or its equalivance. The first 4 points control the curve at one
            side and the last four control the other side.
        num_sample (int): The number of sample points at each Bezeir curve.
    Returns:
        list[ndarray]: A list of 2*num_sample points representing the polygon
        extracted from Bezier curves.
    Warning:
        The points are not guaranteed to be ordered. Please use
        :func:`mmocr.utils.sort_points` to sort points if necessary.
    """
    assert num_sample > 0

    bezier_points = np.asarray(bezier_points)
    assert np.prod(
        bezier_points.shape) == 16, 'Need 8 Bezier control points to continue!'

    bezier = bezier_points.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    u = np.linspace(0, 1, num_sample)

    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
             + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
             + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
             + np.outer(u ** 3, bezier[:, 3])

    # Convert points to polygon
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return points.tolist()
    # return np.array(points.tolist()).reshape(-1, 2)


def sort_points(points):
    """Sort arbitory points in clockwise order. Reference:
    https://stackoverflow.com/a/6989383.
    Args:
        points (list[ndarray] or ndarray or list[list]): A list of unsorted
            boundary points.
    Returns:
        list[ndarray]: A list of points sorted in clockwise order.
    """

    assert is_type_list(points, np.ndarray) or isinstance(points, np.ndarray) \
           or is_2dlist(points)

    points = np.array(points)
    center = np.mean(points, axis=0)

    def cmp(a, b):
        oa = a - center
        ob = b - center

        # Some corner cases
        if oa[0] >= 0 and ob[0] < 0:
            return 1
        if oa[0] < 0 and ob[0] >= 0:
            return -1

        prod = np.cross(oa, ob)
        if prod > 0:
            return 1
        if prod < 0:
            return -1

        # a, b are on the same line from the center
        return 1 if (oa**2).sum() < (ob**2).sum() else -1

    return sorted(points, key=functools.cmp_to_key(cmp))