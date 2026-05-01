import os

base = r'C:\Users\EricHome\Documents\data\LEVIR-CD-256'
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(base, split, 'A')
    names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    out_path = os.path.join(base, 'list', f'{split}.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(names))
    print(f'{split}: {len(names)}개 → {out_path}')