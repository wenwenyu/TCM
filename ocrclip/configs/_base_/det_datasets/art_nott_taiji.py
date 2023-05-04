dataset_type = 'OCRCLIPDataset'
data_root = '/apdcephfs/share_887471/interns/v_fisherwyu/data/mmocr_det_data/art2019'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_ArT_nott.json',
    img_prefix=data_root,
    data_name='art_nott',
    pipeline=None)


train_list = [train]

test_list = [train]