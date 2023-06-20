<div align="center">

  <img src="resources/mmfewshot-logo.png" width="500px"/>

</div>

## Introduction

This reposity is codebase for ACMMM2022 paper ["Exploring Effective Knowledge Transfer for Few-shot Object Detection"](https://arxiv.org/pdf/2210.02021.pdf) .

Codebase is based on MMFewshot. For installing MMFewshot, please refer to [this page](https://github.com/open-mmlab/mmfewshot)

## Requirements

1. MMFewshot installed: https://github.com/open-mmlab/mmfewshot

2. my requirements ( just for your checking, theoretically mmfewshot installed is enough. )

```
python == 3.7.11
torch == 1.7.0
torchvision == 0.8.0
cuda == 10.1
mmfewshot == 0.1.0
```

## Data Preparation

### PASCAL VOC

For PASCAL VOC data, please refer to [GoogleDrive](https://drive.google.com/file/d/1O47rj4RkIKYluUNtAuCLRcGxa_5BsOfQ/view?usp=share_link). Put the archieve under ```data/``` and decompress it.

### MS COCO

### few-shot annotation

MMFewshot using few-shot split file prepared in advance. Download [GoogleDrive](https://drive.google.com/file/d/1EKDb3Kzx8PQ7QriWJL1sDI-Dg5FDqRmI/view?usp=share_link) and also put decompressed folder under ```data/```.

## Model Inference

### MS COCO

| setting | mAP | model |
| --- | --- | --- |
| 1shot | 5.7 | [model](https://drive.google.com/file/d/1bZjoU971P0XZKPVO4Tk68papJe5PTIyr/view?usp=sharing) |
| 2shot | 7.1 | [model](https://drive.google.com/file/d/1QmrXn4XSVjp71AqGVVC7b0qHHLrQZqlu/view?usp=drive_link) |
| 3shot | 8.6 | [model](https://drive.google.com/file/d/14rSqI014ErEEFYhKFaCduWb7DvyCPX8T/view?usp=drive_link) |
| 10shot | 12.5 | - |
| 30shot | 16.7 | [model](https://drive.google.com/file/d/1gIzgNU16pZNfMz5dY2VFFWL1bYVqUrBa/view?usp=drive_link) |

1. download model from link provided and put model under corresponding folder: ```./eval_configs/coco/kshot/```
2. run the following command
   ```
   CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
    ./eval_configs/coco/kshot/config.py \
    ./eval_configs/coco/kshot/best.pth --eval bbox
   ```

## Model Training
- [ ] To be done
