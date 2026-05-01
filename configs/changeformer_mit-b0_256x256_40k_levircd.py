이 아래가 "changeformer_mit-b0_256x256_40k_levircd.py"이고
_base_ = [
    '../_base_/models/changeformer_mit-b0.py', 
    '../common/standard_256x256_40k_levircd.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(pretrained=checkpoint, decode_head=dict(num_classes=2))

# optimizer
optimizer=dict(
    type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))



이 아래가 "changeformer_mit-b1_256x256_40k_levircd.py"이고
_base_ = ['./changeformer_mit-b0_256x256_40k_levircd.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(in_channels=[v * 2 for v in [64, 128, 320, 512]]))

이 아래가 "fix_labels.py"이야
import os
from PIL import Image
import numpy as np
import shutil

root = "data/LEVIR-CD"

for split in ["train", "val", "test"]:
    label_dir = os.path.join(root, split, "label")
    backup_dir = os.path.join(root, split, "label_backup_original")

    if not os.path.exists(label_dir):
        print(f"skip: {label_dir}")
        continue

    if not os.path.exists(backup_dir):
        shutil.copytree(label_dir, backup_dir)
        print(f"backup created: {backup_dir}")
    else:
        print(f"backup already exists: {backup_dir}")

    fixed_count = 0

    for filename in os.listdir(label_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        path = os.path.join(label_dir, filename)
        arr = np.array(Image.open(path).convert("L"))

        fixed = np.zeros_like(arr, dtype=np.uint8)
        fixed[arr > 0] = 1

        if not np.array_equal(arr, fixed):
            Image.fromarray(fixed).save(path)
            fixed_count += 1

    print(f"{split}: fixed {fixed_count} files")

print("done")