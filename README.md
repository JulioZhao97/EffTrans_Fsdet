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

For PASCAL VOC data, please refer to [GoogleDrive](https://drive.google.com/file/d/1O47rj4RkIKYluUNtAuCLRcGxa_5BsOfQ/view?usp=share_link). Put the archieve under data/ and decompress it.

### MS COCO

### few-shot annotation

MMFewshot using few-shot split file prepared in advance. Download [GoogleDrive](https://drive.google.com/file/d/1EKDb3Kzx8PQ7QriWJL1sDI-Dg5FDqRmI/view?usp=share_link) and also put decompressed folder under data/.

## Model Inference

## Model Training
