_base_ = [
    '../_base_/models/swin_transformer_v2/small_256.py',
    '../_base_/datasets/imagenet_bs64_swin_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(img_size=192, window_size=[12, 12, 12, 6]),
)

_base_['train_pipeline'][1]['scale'] = 192  # RandomResizedCrop
_base_['test_pipeline'][1]['scale'] = 219  # ResizeEdge
_base_['test_pipeline'][2]['crop_size'] = 192  # CenterCrop
