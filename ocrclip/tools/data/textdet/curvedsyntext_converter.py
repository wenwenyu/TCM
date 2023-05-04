# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from functools import partial
import functools

import mmcv
import numpy as np
from mmocr.utils.check_argument import is_2dlist, is_type_list

# from mmocr.utils import bezier_to_polygon, sort_points

# The default dictionary used by CurvedSynthText
dict95 = [
    ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.',
    '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=',
    '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[',
    '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
    'z', '{', '|', '}', '~'
]
UNK = len(dict95)
EOS = UNK + 1


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


def digit2text(rec):
    res = []
    for d in rec:
        assert d <= EOS
        if d == EOS:
            break
        if d == UNK:
            print('Warning: Has a UNK character')
            res.append('口')  # Or any special character not in the target dict
        res.append(dict95[d])
    return ''.join(res)


def modify_annotation(ann, num_sample, start_img_id=0, start_ann_id=0):
    ann['text'] = digit2text(ann.pop('rec'))
    # Get hide egmentation points
    polygon_pts = bezier_to_polygon(ann['bezier_pts'], num_sample=num_sample)
    seg = np.asarray(sort_points(polygon_pts)).reshape(1, -1).tolist()
    ann['segmentation'] = seg
    ann['image_id'] += start_img_id
    if not (len(seg[0]) >=8 and len(seg[0])%2==0):
        with open('error_img_id.txt', 'a+') as f:
            f.write(f'{ann["image_id"]}\n')
    ann['id'] += start_ann_id
    return ann


def modify_image_info(image_info, path_prefix, start_img_id=0):
    image_info['file_name'] = osp.join(path_prefix, image_info['file_name'])
    image_info['id'] += start_img_id
    return image_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CurvedSynText150k to COCO format')
    parser.add_argument('root_path', help='CurvedSynText150k  root path')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument(
        '-n',
        '--num-sample',
        type=int,
        default=4,
        help='Number of sample points at each Bezier curve.')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def convert_annotations(data,
                        path_prefix,
                        num_sample,
                        nproc,
                        start_img_id=0,
                        start_ann_id=0):
    modify_image_info_with_params = partial(
        modify_image_info, path_prefix=path_prefix, start_img_id=start_img_id)
    modify_annotation_with_params = partial(
        modify_annotation,
        num_sample=num_sample,
        start_img_id=start_img_id,
        start_ann_id=start_ann_id)
    if nproc > 1:
        data['annotations'] = mmcv.track_parallel_progress(
            modify_annotation_with_params, data['annotations'], nproc=nproc)
        data['images'] = mmcv.track_parallel_progress(
            modify_image_info_with_params, data['images'], nproc=nproc)
    else:
        data['annotations'] = mmcv.track_progress(
            modify_annotation_with_params, data['annotations'])
        data['images'] = mmcv.track_progress(
            modify_image_info_with_params,
            data['images'],
        )
    data['categories'] = [{'id': 1, 'name': 'text'}]
    return data


def main():
    args = parse_args()
    root_path = args.root_path
    out_dir = args.out_dir if args.out_dir else root_path
    mmcv.mkdir_or_exist(out_dir)

    anns = mmcv.load(osp.join(root_path, 'train1.json'))
    data1 = convert_annotations(anns, 'syntext_word_eng', args.num_sample,
                                args.nproc)

    # Get the maximum image id from data1
    start_img_id = max(data1['images'], key=lambda x: x['id'])['id'] + 1
    start_ann_id = max(data1['annotations'], key=lambda x: x['id'])['id'] + 1
    anns = mmcv.load(osp.join(root_path, 'train2.json'))
    data2 = convert_annotations(
        anns,
        'emcs_imgs',
        args.num_sample,
        args.nproc,
        start_img_id=start_img_id,
        start_ann_id=start_ann_id)

    data1['images'] += data2['images']
    data1['annotations'] += data2['annotations']
    mmcv.dump(data1, osp.join(out_dir, 'instances_training.json'))


if __name__ == '__main__':
    '''
    python tools/data/textdet/curvedsyntext_converter.py /data/wwyu/mmocr_det_data/syntext150k --nproc 38
    '''
    main()
