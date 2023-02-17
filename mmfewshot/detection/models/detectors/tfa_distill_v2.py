# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage_distill_v2 import TwoStageDetectorDistillV2


@DETECTORS.register_module()
class TFADistillV2(TwoStageDetectorDistillV2):
    """Implementation of `TFA <https://arxiv.org/abs/2003.06957>`_"""
