## Turning a CLIP Model into a Scene Text Spotter
This repository is build upon [mmrotate 1.0.0](https://github.com/open-mmlab/mmrotate/tree/v1.0.0rc1).


### DOTA Dataset
DOTA dataset, can be downloaded from [here](https://github.com/open-mmlab/mmrotate/tree/main/tools/data).


## Usage


### Environment
- cuda 12.1
- torch=2.1.0
- torchvision=0.16.0
- mmcv-full=2.1.0
- mmdet=3.2.0
- mmrotate=1.0.0rc1
- clip=1.0

The code is based on mmrotate & CLIP. Please first install the `mmcv-full` and `mmdet` following the official guidelines ([mmrotate](https://mmrotate.readthedocs.io/en/latest/install.html)), then install [CLIP](https://github.com/openai/CLIP).

### Dataset
- Please following the mmrotate official guidelines to prepare the [datasets](https://mmrotate.readthedocs.io/en/latest/get_started.html) accordingly. 

- Configure the dataset path in [`CLIP/config_TCM/TCM_dota.py`](CLIP/config_TCM/TCM_dota.py).

### Pre-trained CLIP Models

- Download the pre-trained CLIP models (`RN50.pt`) and save them to the `pretrained` folder.
- Configure the pre-trained CLIP models path in [config file](CLIP/config_TCM/rotated-fcosTCM-le90_r50_fpn_1x_dota.py) as

```python
# model settings
ckpt_path = '/xxx/RN50.pt'
```

### Training & Evaluation 


To finetune the TCM model based on pretrained RN50.pt, please set the `ckpt_path`, then run:

```
python ./tools/train.py CLIP/config_TCM/rotated-fcosTCM-le90_r50_fpn_1x_dota.py --work-dir ./work_dirs/r-fcos-tcm
```

To evaluate the performance with checkpoint, run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh work_dirs/r-fcos-tcm/rotated-fcosTCM-le90_r50_fpn_1x_dota.py work_dirs/r-fcos-tcm/epoch_12.pth 4
```

### Results



| Method | Data | AP50 (single scale) | Model |
|--------|------|-----------|--------|
| TCM-rotated-FCOS | DOTA |    75.1%       |    [config](CLIP/config_TCM/rotated-fcosTCM-le90_r50_fpn_1x_dota.py) \| [log](https://mega.nz/file/TAgxyZ4B#sEUa5IJM-N3qSOl-0iFogONgPtgRtccAqUGEEiTP-ZY) \| [weights](https://mega.nz/file/TU4CHABb#8BZJXm_A0qiFrW8smRC52DhOFUNAlrw16L1yLnz6ZzY)   |  
| TCM-rotated-ATSS | DOTA |    76.1%       |   [config](CLIP/config_TCM/rotated-atssTCM-le90_r50_fpn_1x_dota.py) \| [log](https://mega.nz/file/SQhEjZ5S#QNmJgq5PTHKgsGhFvqP6AhcJJs3MbO3uhLH6MJ3AaVY) \| [weights](https://mega.nz/file/TU4CHABb#8BZJXm_A0qiFrW8smRC52DhOFUNAlrw16L1yLnz6ZzY)    |                   
| TCM-rotaed-retinanet | DOTA |    70.99%       |  [config](CLIP/config_TCM/rotated-retinanetTCM-rbox-le90_r50_fpn_1x_dota.py) \| [log](https://mega.nz/file/OQ4kRaiB#z4sO6L_ViJC05o_zlw8_F-hovM8WCcGWr1bI87C93YE) \| [weights](https://mega.nz/file/TU4CHABb#8BZJXm_A0qiFrW8smRC52DhOFUNAlrw16L1yLnz6ZzY)     |                   


### Cites
If you find this project helpful for your research, please consider citing the paper

```
@inproceedings{Yu2023TurningAC,
  title={Turning a CLIP Model into a Scene Text Detector},
  author={Wenwen Yu and Yuliang Liu and Wei Hua and Deqiang Jiang and Bo Ren and Xiang Bai},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

@article{Yu2024TurningAC,
  title={Turning a CLIP Model into a Scene Text Spotter},
  author={Wenwen Yu and Yuliang Liu and Xingkui Zhu and Haoyu Cao and Xing Sun and Xiang Bai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```

### Licence
This project is under the CC-BY-NC 4.0 license.


### Acknowledges
The project partially based on [MMRotate](https://github.com/open-mmlab/mmrotate), [CLIP](https://github.com/openai/CLIP), [DenseCLIP](https://github.com/raoyongming/DenseCLIP). Thanks for their great works.