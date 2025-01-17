# model settings
model = dict(
    type='SOLOv2',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(
            type='DCNv2',
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5
    ),
    bbox_head=dict(
        type='SOLOv2Head',
        # change num_classes
        num_classes=2,  # 1+1 foreground+background
        in_channels=256,
        stacked_convs=4,
        use_dcn_in_tower=True,
        type_dcn='DCNv2',
        seg_feat_channels=512,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
        # loss_ins=dict(
        #     type='DiceLoss',
        #     use_sigmoid=True,
        #     loss_weight=1.0),  # loss_weight=3.0),
        loss_ins=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=6.0),
        # loss_ins=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=True,
        #     loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
    ),
    mask_feat_head=dict(
        type='MaskFeatHead',
        in_channels=256,
        out_channels=128,
        start_level=0,
        end_level=3,
        num_classes=256,
        conv_cfg=dict(type='DCNv2'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    ),
)
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)
# dataset settings
dataset_type = 'SyntheticsDataset'
data_root = '/ldap_home/jincheng.lyu/data/product_segmentation/synthetics_cocoformat_crop/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(320, 192), (320, 184), (320, 176),
                    (192, 192), (184, 320), (176, 320), (320, 320)],           
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

ann_root = '/ldap_home/jincheng.lyu/project/SOLO/data/pinctada/annotations_clsag_crop/'
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train.json',
        ann_file=ann_root + 'instances_train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'instances_test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[27//4, 33//4])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 50//4
device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/solov2_release_x101_dcn_fpn_2gpu_3x_6lambda_bothfocalloss_Tdiv4_rlemask_clsag_fixhead_fixcate_crop'
load_from = None
resume_from = './work_dirs/solov2_release_x101_dcn_fpn_2gpu_3x_6lambda_bothfocalloss_Tdiv4_rlemask_clsag_fixhead_fixcate_crop/epoch_5.pth'
workflow = [('train', 1)]
