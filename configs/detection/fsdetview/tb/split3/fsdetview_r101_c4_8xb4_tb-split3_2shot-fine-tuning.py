img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='GenerateMask', target_size=(224, 224)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = 'data/VOCdevkit_tb/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='NWayKShotDataset',
        num_support_ways=15,
        num_support_shots=1,
        one_support_shot_per_image=False,
        num_used_support_shots=30,
        save_dataset=True,
        dataset=dict(
            type='FewShotVOCDefaultDatasetTB',
            ann_cfg=[dict(method='FSDetView', setting='SPLIT3_2SHOT', task='TASK0')],
            img_prefix='data/VOCdevkit_tb/',
            multi_pipelines=dict(
                query=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Resize', img_scale=(1000, 600), keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ],
                support=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='GenerateMask', target_size=(224, 224)),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]),
            classes='ALL_CLASSES_SPLIT3',
            use_difficult=False,
            instance_wise=False,
            dataset_name='query_support_dataset',
            num_novel_shots=2,
            num_base_shots=2)),
    val=dict(
        type='FewShotVOCDatasetTB',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/VOCdevkit_tb/VOC2007/ImageSets/Main/few-shot/test.txt')
        ],
        img_prefix='data/VOCdevkit_tb/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1000, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes='ALL_CLASSES_SPLIT3'),
    test=dict(
        type='FewShotVOCDatasetTB',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/VOCdevkit_tb/VOC2007/ImageSets/Main/few-shot/test.txt')
        ],
        img_prefix='data/VOCdevkit_tb/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1000, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        test_mode=True,
        classes='ALL_CLASSES_SPLIT3'),
    model_init=dict(
        copy_from_train_dataset=True,
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotVOCDatasetTB',
        ann_cfg=None,
        img_prefix='data/VOCdevkit_tb/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='GenerateMask', target_size=(224, 224)),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        use_difficult=False,
        instance_wise=True,
        num_novel_shots=None,
        classes='ALL_CLASSES_SPLIT3',
        dataset_name='model_init_dataset'))
evaluation = dict(
    interval=500,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[30000, 40000])
runner = dict(type='IterBasedRunner', max_iters=10000)
norm_cfg = dict(type='BN', requires_grad=False)
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
model = dict(
    type='FSDetView',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(
        type='ResNetWithMetaConv',
        depth=101,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=2,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=512,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            scale_major=False,
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='FSDetViewRoIHead',
        shared_head=dict(
            type='MetaRCNNResLayer',
            pretrained='open-mmlab://detectron2/resnet101_caffe',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        bbox_head=dict(
            type='MetaBBoxHead',
            with_avg_pool=False,
            roi_feat_size=1,
            in_channels=4096,
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),
            num_meta_classes=20,
            meta_cls_in_channels=2048,
            with_meta_cls_loss=True,
            loss_meta=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DepthWiseCorrelationAggregator',
                    in_channels=2048,
                    out_channels=1024,
                    with_fc=True),
                dict(
                    type='DifferenceAggregator',
                    in_channels=2048,
                    out_channels=1024,
                    with_fc=True)
            ],
            init_cfg=[
                dict(
                    type='Normal',
                    layer=['Conv1d', 'Conv2d', 'Linear'],
                    mean=0.0,
                    std=0.001),
                dict(type='Normal', layer=['BatchNorm1d'], mean=1.0, std=0.02)
            ])),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=12000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100)),
    frozen_parameters=[
        'backbone', 'shared_head', 'rpn_head', 'aggregation_layer'
    ])
checkpoint_config = dict(interval=10000)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/tb/fsdetview/split3/fsdetview_r101_c4_8xb4_tb-split3_base-training/iter_10000.pth'
resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = True
seed = 42
work_dir = './work_dirs/tb/fsdetview/split3/fsdetview_r101_c4_8xb4_tb-split3_2shot-fine-tuning'
gpu_ids = range(0, 4)