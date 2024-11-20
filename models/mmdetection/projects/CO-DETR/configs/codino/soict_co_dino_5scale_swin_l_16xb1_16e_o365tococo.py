_base_ = ['soict_co_dino_5scale_r50_8xb2_1x_coco.py']

pretrained = 'pretrained_weights/swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'pretrained_weights/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa
# resume = 'work_dirs/soict_co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_18.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(720, 1280), keep_ratio=True),
    dict(type='PackDetInputs')
]

data_root = 'data/soict_vehicle/'
metainfo = {
    'classes': ('0', '1', '2', '3'),
}
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size=2, num_workers=1, 
    dataset=dict(
        ann_file='annotations/augmented_sample_1.json',
        data_prefix=dict(img='augmented_sample_1_images/'),
        pipeline=train_pipeline,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
    )
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(720, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=2, num_workers=1, 
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val_all_coco.json',
        data_prefix=dict(img='val_all_images/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

test_dataloader = dict(
    batch_size=10, num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/public_test_coco.json',
        data_prefix=dict(img='public_test_images/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val_all_coco.json',
    metric='bbox',
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/public_test_coco.json',
    metric='bbox',
    format_only=False)

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 32
train_cfg = dict(max_epochs=max_epochs, val_interval=32)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
