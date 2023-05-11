_base_ = [
    '../_base_/models/swin_transformer_v2/small_256.py',
    '../_base_/datasets/imagenet_bs64_swin_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(img_size=192, window_size=[12, 12, 12, 6]),
    head=dict(num_classes=7),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root='C:/Users/philo/OneDrive/Desktop/model/IMAGE BACKBONE',
        ann_file='C:/Users/philo/OneDrive/Desktop/model/IMAGE BACKBONE/train_ann.txt',
        data_prefix='ISIC2018_Task3_Training_Input'),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root='C:/Users/philo/OneDrive/Desktop/model/IMAGE BACKBONE',
        ann_file='C:/Users/philo/OneDrive/Desktop/model/IMAGE BACKBONE/valid_ann.txt',
        data_prefix='ISIC2018_Task3_Validation_Input'),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# dataset settings
data_preprocessor = dict(num_classes=7)

_base_['train_pipeline'][1]['scale'] = 192  # RandomResizedCrop
_base_['test_pipeline'][1]['scale'] = 219  # ResizeEdge
_base_['test_pipeline'][2]['crop_size'] = 192  # CenterCrop

work_dir = './work_dirs/isiv2018'
load_from = 'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-small-w8_3rdparty_in1k-256px_20220803-b01a4332.pth'
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
val_cfg = dict()
test_cfg = dict()

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=10,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=1)
]