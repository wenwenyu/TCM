# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 5/1/22 7:17 PM

import functools
import mmcv
import os
from tqdm import tqdm
import math
import numpy as np
import io
import os.path as osp
import zipfile
import os.path as osp
import zipfile
def sort_points(points):
    """Sort arbitory points in clockwise order. Reference:
    https://stackoverflow.com/a/6989383.
    Args:
        points (list[ndarray] or ndarray or list[list]): A list of unsorted
            boundary points.
    Returns:
        list[ndarray]: A list of points sorted in clockwise order.
    """

    # assert is_type_list(points, np.ndarray) or isinstance(points, np.ndarray) \
    #        or is_2dlist(points)

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

def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
               / (xy_list[index, 0] - xy_list[first_v, 0] + 1e-4)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
            xy_list[second_v, 0] - xy_list[fourth_v, 0] + 1e-4)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list

def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
        [int(points[0]) , int(points[1])],
        [int(points[2]) , int(points[3])],
        [int(points[4]) , int(points[5])],
        [int(points[6]) , int(points[7])]
    ]
    edge = [
        ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
        ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
        ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
        ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory>0:
        print(
            "Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")
        return False
        # raise Exception("Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")
    return True


res = mmcv.load('/home/wwyu/code/OCRCP/ic17best.pkl')

out_dir = '/home/wwyu/code/OCRCP/res_ic17_4.zip'
tmp_folder = out_dir.replace('.zip', '')

if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)

for pred in tqdm(res):
    filename = pred['filename'] # e.g. /abs/path/ts_img_id.png
    file_id = os.path.basename(filename).split('.')[0].split('_')[-1]
    boundary_result = pred['boundary_result']
    file_name = f'res_img_{file_id}.txt'
    file_path = os.path.join(tmp_folder, file_name)
    with io.open(file_path, 'w', newline='\r\n') as f:
        for idx, pred_det in enumerate(boundary_result):
            # x1,...,x8,score
            # points = pred_det[:8]
            # if not validate_clockwise_points(points):
            #     points = reorder_vertexes(points)
            #     print('not clockwise')
            # sort_pred_det =[]
            pred_det_final = []
            # sort_pred_det.extend(points)
            # sort_pred_det.append(pred_det[-1])
            for x_id, x in enumerate(pred_det):
                if x_id != 8:
                    x_final = f'{int(x)}'
                else:
                    x_final = f'{x:.2}'
                pred_det_final.append(x_final)
            pred_line = ','.join(pred_det_final)
            # if idx == len(boundary_result)-1:
            #     f.write(f'{pred_line}')
            # else:
            f.write('{}{}'.format(pred_line, '\n'))  # \r\n not \n

    z = zipfile.ZipFile(out_dir, 'a', zipfile.ZIP_DEFLATED)
    z.write(file_path, file_name)
    z.close()

