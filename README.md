## Turning a CLIP Model into a Scene Text Detector
This repository is build upon [mmocr 0.4.0](https://github.com/open-mmlab/mmocr/tree/0.x).


### NightTime-ArT Dataset
NightTime-ArT dataset, collected from ArT, can be downloaded from [here](https://drive.google.com/file/d/1v3CshPqlvhpnK1_MKwqqkWJDikKl_g4Y).


## Usage


### Environment
- cuda 11.1
- torch=1.8.0
- torchvision=0.9.0
- timm=0.4.12
- mmcv-full=1.3.17
- mmseg=0.20.2
- mmdet=2.19.1
- mmocr=0.4.0

The code is based on mmocr. Please first install the `mmcv-full` and `mmocr` following the official guidelines ([mmocr](https://github.com/open-mmlab/mmocr)).

### Dataset
- Please following the mmocr official guidelines to prepare the [datasets](https://mmocr.readthedocs.io/en/v0.4.1/datasets/det.html) accordingly. 

- Configure the dataset path in [`ocrclip/configs/_base_/det_datasets`](ocrclip/configs/_base_/det_datasets).

### Pre-trained CLIP Models

- Download the pre-trained CLIP models (`RN50.pt`) and save them to the `pretrained` folder.
- Configure the pre-trained CLIP models path in config file as

```python
model = dict(
    pretrained='xxx/ocrclip/pretrained/RN50.pt',
    )
```

### Pretraining & Training & Evaluation 

To pretrain the TCM model on SynthText/Synth150k, please configure the corresponding dataset path, then run:

```
bash dist_train.sh configs/textdet/xxnet/xxx.py 8
```

To finetune the TCM model based on pretrained model, please configure the `load_from` to the pretrained checkpoint path, then run:

```
bash dist_train.sh configs/textdet/xxnet/xxx.py 8
```

To evaluate the performance with checkpoint, run:

```
bash dist_test.sh configs/textdet/xxnet/xxx.py /path/to/checkpoint 1 --eval hmean-iou
```

### Results



| Method | Data | F-measure | Model |
|--------|------|-----------|--------|
| TCM-DB | TD |    88.8%       |    [config](ocrclip/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_td_ranger_post_taiji.py) [weights](https://mega.nz/file/daZWnYQI#XTQbvp86rxf-zIoQKQwVcXeUnGNqj4ADm1OijQKgEMM)   |  
| TCM-DB | IC15 |    88.8%       |   [config](ocrclip/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_1200e_ft_gen_ic15_adam_taiji.py) [weights](https://mega.nz/file/cDQ1RASb#k5IOBtv12legGQPFCBW4-7e8SuD9WXcX4uoTE4Z9hpA)    |                   
| TCM-DB | CTW |    85.1%       |  [config](ocrclip/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_32_1200e_ft_ctw_adamw_taiji.py)      |            
| TCM-DB | TT |    85.9%       |    [config](ocrclip/configs/textdet/dbnet/clip_db_r50_fpnc_prompt_gen_vis_32_1200e_ft_tt_adamw_taiji.py)    |            


### TODO
- [ ] Add FastTCM
- [ ] Migration from mmocr 0.4.0 to mmocr 1.0.0
- [ ] Refactor and clean code
- [ ] Release domain adaptation setting

### Cites
If you find this project helpful for your research, please consider citing the paper

```
@inproceedings{Yu2023TurningAC,
  title={Turning a CLIP Model into a Scene Text Detector},
  author={Wenwen Yu and Yuliang Liu and Wei Hua and Deqiang Jiang and Bo Ren and Xiang Bai},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

### Licence
This project is under the CC-BY-NC 4.0 license. See `LICENSE` for more details.


### Acknowledges
The project partially based on [MMOCR](https://github.com/open-mmlab/mmocr), [CLIP](https://github.com/openai/CLIP), [DenseCLIP](https://github.com/raoyongming/DenseCLIP). Thanks for their great works.
