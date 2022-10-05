dataset_type = 'CocoDataset'
img_size = (992, 736)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672), (992, 800)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type="LabelNoiseBundle"),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'noise_labels', 'anno_ids'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes = ['movable-objects', 'boat', 'car', 'dock', 'jetski', 'lift']
classes = ['dice1', 'dice2', 'dice3', 'dice4', 'dice5', 'dice6']
# classes = ['cells', 'Platelets', 'RBC', 'WBC']
# classes = ['spaces', 'space-empty', 'space-occupied']

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        adaptive_repeat_times=True,
        times=1,
        dataset=dict(
            type='LabelNoiseCocoDataset',
            ann_file='/mnt/ssd2/sc_datasets_det/d6-dice/annotations/instances_train_16_1_noise.json',
            img_prefix='/mnt/ssd2/sc_datasets_det/d6-dice/images/train',
            classes=classes,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file='/mnt/ssd2/sc_datasets_det/d6-dice/annotations/instances_val_100.json',
        img_prefix='/mnt/ssd2/sc_datasets_det/d6-dice/images/val',
        test_mode=True,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/mnt/ssd2/sc_datasets_det/d6-dice/annotations/instances_test.json',
        img_prefix='/mnt/ssd2/sc_datasets_det/d6-dice/images/test',
        test_mode=True,
        classes=classes,
        pipeline=test_pipeline))

model = dict(
    type='ATSS',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(2, 3, 4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=64,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='ATSSHeadWithLossDynamics',
        num_classes=len(classes),
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction="none"
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0, reduction="none"),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, reduction="none")),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

evaluation = dict(interval=5, metric='mAP', save_best='mAP')
optimizer = dict(
    type='SGD',
    lr=0.008,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict()
# lr_config = dict(
#     policy='ReduceLROnPlateau',
#     metric='mAP',
#     patience=5,
#     iteration_patience=600,
#     interval=1,
#     min_lr=0.000008,
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=1.0 / 3)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[50, 65])

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
runner = dict(type='EpochBasedRunner', max_epochs=40)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'output'
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-atss.pth'
resume_from = None
workflow = [('train', 1)]
custom_hooks = [
    dict(type='SaveLossDynamicsHook')
]
