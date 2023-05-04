# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 1/18/22 8:38 PM

import mmcv

def merge(out_json_name, data_names, data_root):
    # splits = 'test_gts'
    # train_data_names = ['MLT17', 'MLT19', 'TextOCR', 'ICDAR2019-ArT', 'COCOTEXT', 'ICDAR2019-LSVT-0', 'ICDAR2019-LSVT-1']
    # test_data_names = ['MLT17', 'TextOCR', 'ICDAR2019-ArT', 'COCOTEXT', 'ICDAR2019-LSVT-0', 'ICDAR2019-LSVT-1']
    # out_json_name = f'{data_root}/train_gts/gt_jsons/instances_all.json'

    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    cat = dict(id=1, name='text')
    out_json['categories'].append(cat)
    for data_name in data_names:
        train_ann_file1 = f'{data_root}/{data_name}'
        data = mmcv.load(train_ann_file1)
        imgs_tmp = data['images']
        anns_tmp = data['annotations']
        for img_info in imgs_tmp:
            img_id_tmp = img_info['id']
            img_info['id'] = img_id
            out_json['images'].append(img_info)
            for anno_info in anns_tmp:
                if anno_info['image_id'] == img_id_tmp:
                    anno_info['image_id'] = img_id
                    anno_info['id'] = ann_id
                    out_json['annotations'].append(anno_info)
                    ann_id += 1
            img_id += 1

    mmcv.dump(out_json, out_json_name)
    print(img_id, ann_id)

if __name__ == '__main__':
    # data_root = '/apdcephfs/share_887471/common/ocr_benchmark/mmocr_det_data/Text_at_Night'
    data_root = '/apdcephfs/share_887471/common/ocr_benchmark/mmocr_det_data/icdar2017'
    data_names = ['instances_training_en.json', 'instances_val_en.json']
    out_name = 'instances_training_val_en.json'
    out_json_name = f'{data_root}/{out_name}'
    merge(out_json_name, data_names, data_root)
