# dataset settings
classes = ('daytime', 'nighttime')
num_classes = len(classes)
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=224, edge='short', backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=224, edge='short', backend='pillow'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='data/soict_imagenet',
        split='train',
        pipeline=train_pipeline,
        classes=classes),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='data/soict_imagenet',
        split='val',
        pipeline=test_pipeline,
        classes=classes),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1,))

# If you want standard test, please manually configure the test dataset
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=4,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_prefix='test',
        data_root='data/soict_imagenet',
        pipeline=test_pipeline,
        classes=classes),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator
