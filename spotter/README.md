## Turning a CLIP Model into a Scene Text Spotter
This repository is build upon [AdelaiDet](https://github.com/aim-uofa/AdelaiDet).

## Usage

### Environment

+ CUDA 11.3
+ Python 3.8
+ PyTorch 1.10.1
+ Official Pre-Built Detectron2

### Installation

Please refer to the **Installation** section of AdelaiDet: [README.md](https://github.com/aim-uofa/AdelaiDet/blob/master/README.md). 

If you have not installed Detectron2, following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md). 


### Datasets

Please following the offlical guide provided by [AdelaiDet](https://github.com/aim-uofa/AdelaiDet/blob/master/datasets/README.md) to prepare the datasets.

After that, download [polygonal annotations](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/ES4aqkvamlJAgiPNFJuYkX4BLo-5cDx9TD_6pnMJnVhXpw?e=tu9D8t), along with [evaluation files](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/xiz102_ucsd_edu/Ea5oF7VFoe5NngUoPmLTerQBMdiVUhHcx2pPu3Q5p3hZvg?e=2NJNWh) and extract them under `datasets` folder provided by TESTR.

### Training

You can train the model by putting pretrained weights in `weights` folder.

Example commands:

```bash
python tools/train_net.py --config-file /path/to/config --num-gpus 8
```

Configuration files can be found in `configs`.


### Evaluation

```bash
python tools/train_net.py --config-file /path/to/config --eval-only MODEL.WEIGHTS /path/to/model
```

## Citation

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
The project partially based on [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), [CLIP](https://github.com/openai/CLIP), [DenseCLIP](https://github.com/raoyongming/DenseCLIP), [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), [TESTR](https://github.com/mlpc-ucsd/TESTR). Thanks for their great works.

