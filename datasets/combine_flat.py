import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs from flat folders')
parser.add_argument('--fold_A', type=str, required=True, help='blurry images folder')
parser.add_argument('--fold_B', type=str, required=True, help='sharp images folder')
parser.add_argument('--fold_AB', type=str, required=True, help='output folder')
args = parser.parse_args()

os.makedirs(args.fold_AB, exist_ok=True)

img_list = sorted(os.listdir(args.fold_A))
print(f'Found {len(img_list)} images')

for name in img_list:
    path_A = os.path.join(args.fold_A, name)
    path_B = os.path.join(args.fold_B, name)
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
        im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(os.path.join(args.fold_AB, name), im_AB)

print('Done!')