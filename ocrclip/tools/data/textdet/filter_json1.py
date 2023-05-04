# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 1/18/22 8:38 PM

# filter file if text bounding box points is <=8 and %2!=0

import mmcv
from tqdm import tqdm

def merge(data_names, data_root):

    for data_name in data_names:
        train_ann_file1 = f'{data_root}/{data_name}'
        data = mmcv.load(train_ann_file1)
        imgs_tmp = data['images']
        anns_tmp = data['annotations']
        for img_info in imgs_tmp:
            img_id_tmp = img_info['id']
            for anno_info in anns_tmp:
                if anno_info['image_id'] == img_id_tmp:
                    # bbox = anno_info['bbox']
                    seg = anno_info['segmentation'][0]
                    if not (len(seg) >=8 and len(seg)%2==0):
                        print(f"data:{data_name} img:{img_info['file_name']}, annid:{anno_info['id']}")
    print('done')


if __name__ == '__main__':
    # data_root = '/apdcephfs/share_887471/common/ocr_benchmark/mmocr_det_data/Text_at_Night'
    # data_root = '/apdcephfs/share_887471/common/ocr_benchmark/mmocr_det_data/art2019'
    data_names = ['instances_ArT_noctw.json', 'instances_ArT_nott.json']
    data_root = '/data/wwyu/mmocr_det_data/syntext150k'
    data_names = ['instances_training1.json']
    merge(data_names, data_root)

