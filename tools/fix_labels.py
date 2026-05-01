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