import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# 경로 설정
data_root = r'C:\Users\EricHome\Documents\data\LEVIR-CD-256'
checkpoint_root = r'C:\Users\EricHome\Documents\BIT_CD\checkpoints\BIT_LEVIR'
output_dir = r'C:\Users\EricHome\Documents\BIT_CD\vis_portfolio'
os.makedirs(output_dir, exist_ok=True)

# 테스트 이미지 목록
list_path = os.path.join(data_root, 'list', 'test.txt')
img_names = open(list_path).read().strip().split('\n')

# 모델 로딩
import torch
import sys
sys.path.insert(0, r'C:\Users\EricHome\Documents\BIT_CD')
from models.networks import define_G
from argparse import Namespace
import torchvision.transforms as T

args = Namespace(net_G='base_transformer_pos_s4_dd8_dedim8', gpu_ids=[0])
net = define_G(args).cuda()
ckpt = torch.load(os.path.join(checkpoint_root, 'best_ckpt.pt'))
net.load_state_dict(ckpt['model_G_state_dict'])
net.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# 샘플 랜덤 선택
random.seed(42)
samples = random.sample(img_names, min(20, len(img_names)))

for name in samples:
    A_path = os.path.join(data_root, 'test', 'A', name)
    B_path = os.path.join(data_root, 'test', 'B', name)
    L_path = os.path.join(data_root, 'test', 'label', name)

    imgA = Image.open(A_path).convert('RGB')
    imgB = Image.open(B_path).convert('RGB')
    label = np.array(Image.open(L_path)) // 255

    with torch.no_grad():
        tA = transform(imgA).unsqueeze(0).cuda()
        tB = transform(imgB).unsqueeze(0).cuda()
        pred = net(tA, tB)
        pred_mask = pred.argmax(1).squeeze().cpu().numpy()

    # GT, Pred 마스크를 RGB로 변환 (흰=changed, 검=unchanged)
    def mask_to_rgb(mask):
        img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        img[mask == 1] = [255, 255, 255]
        return Image.fromarray(img)

    gt_img   = mask_to_rgb(label)
    pred_img = mask_to_rgb(pred_mask)

    # 4분할 캔버스 생성
    W, H = 256, 256
    pad = 4
    label_h = 24
    canvas_w = W * 4 + pad * 5
    canvas_h = H + pad * 2 + label_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (40, 40, 40))

    titles = ['Before (A)', 'After (B)', 'GT Mask', 'Pred Mask']
    imgs   = [imgA, imgB, gt_img, pred_img]

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        x = pad + i * (W + pad)
        canvas.paste(img, (x, pad))
        draw.text((x + 4, pad + H + 4), title, fill=(200, 200, 200), font=font)

    out_path = os.path.join(output_dir, name.replace('.png', '_vis.jpg'))
    canvas.save(out_path, quality=95)
    print(f'저장: {out_path}')

print(f'\n완료! {output_dir} 에서 확인하세요.')