# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 1/18/22 8:38 PM


# filter file if text bounding box points is <=8 and %2!=0
import mmcv

def merge(out_json_name, data_name, data_root):

    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    cat = dict(id=1, name='text')
    out_json['categories'].append(cat)

    train_ann_file1 = f'{data_root}/{data_name}'
    data = mmcv.load(train_ann_file1)
    imgs_tmp = data['images']
    anns_tmp = data['annotations']
    for img_info in imgs_tmp:
        img_id_tmp = img_info['id']
        is_bad = False
        for anno_info in anns_tmp:
            if anno_info['image_id'] == img_id_tmp:
                seg = anno_info['segmentation'][0]
                if not (len(seg) >=8 and len(seg)%2==0):
                    is_bad=True
                    #     print(f"data:{data_name} img:{img_info['file_name']}, annid:{anno_info['id']}")
                    break
        if is_bad:continue
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
    data_root = '/apdcephfs/share_887471/common/ocr_benchmark/mmocr_det_data/art2019'
    data_names = ['instances_ArT_noctw.json', 'instances_ArT_nott.json']
    out_json_names = [f'{data_root}/instances_ArT_noctw1.json', f'{data_root}/instances_ArT_nott1.json',]
    for idx, daname in enumerate(data_names):
        merge(out_json_names[idx], daname, data_root)
