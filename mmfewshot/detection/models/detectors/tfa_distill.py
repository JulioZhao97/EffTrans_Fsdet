# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage_distill import TwoStageDetectorDistill


@DETECTORS.register_module()
class TFADistill(TwoStageDetectorDistill):
    """Implementation of `TFA <https://arxiv.org/abs/2003.06957>`_"""
