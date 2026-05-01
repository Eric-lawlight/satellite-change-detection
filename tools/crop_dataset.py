import os
from PIL import Image

src = r'C:\Users\EricHome\Documents\data\LEVIR-CD'
dst = r'C:\Users\EricHome\Documents\data\LEVIR-CD-256'
patch_size = 256
stride = 256

for split in ['train', 'val', 'test']:
    for folder in ['A', 'B', 'label']:
        os.makedirs(os.path.join(dst, split, folder), exist_ok=True)

    img_names = sorted(os.listdir(os.path.join(src, split, 'A')))
    count = 0
    for name in img_names:
        stem = name.replace('.png', '')
        imgs = {}
        for folder in ['A', 'B', 'label']:
            imgs[folder] = Image.open(os.path.join(src, split, folder, name))
        
        W, H = imgs['A'].size
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                box = (x, y, x + patch_size, y + patch_size)
                patch_name = f'{stem}_{y}_{x}.png'
                for folder in ['A', 'B', 'label']:
                    imgs[folder].crop(box).save(
                        os.path.join(dst, split, folder, patch_name))
                count += 1
    print(f'{split}: {count}개 패치 생성 완료')

print('크롭 완료!')