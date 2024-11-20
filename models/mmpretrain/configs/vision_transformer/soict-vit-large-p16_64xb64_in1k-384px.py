_base_ = [
    '../_base_/models/soict-vit-large-p16.py',
    '../_base_/datasets/soict_imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

load_from = 'pretrained_weights/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth'
# resume = './work_dirs/soict-vit-large-p16_64xb64_in1k-384px/epoch_3.pth'

# model setting
model = dict(backbone=dict(img_size=384))

# dataset setting
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='short', backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='short', backend='pillow'),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=4)
val_cfg = dict()
test_cfg = dict()

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))
