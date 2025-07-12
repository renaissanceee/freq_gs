import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path
import torch
from scene.visualize_utils import read_json, draw_oriented_arrows, draw_mag, draw_oriented_arrows_per_patch

def visualize_per_patch_freq(image_path, output_path, blur, patch_size=20):
    image_bgr = cv2.imread(image_path)
    if blur:
        image_bgr = cv2.GaussianBlur(image_bgr, (5, 5), sigmaX=1.0) # blur
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    n_patches_y = h // patch_size
    n_patches_x = w // patch_size

    # f_magnitude = torch.zeros((n_patches_y, n_patches_x))
    # f_orientation = torch.zeros((n_patches_y, n_patches_x))
    f_magnitude = np.zeros((n_patches_y, n_patches_x))
    f_orientation = np.zeros((n_patches_y, n_patches_x))
    full_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n_patches_y):  # edge px not involved
        for j in range(n_patches_x):
            patch = gray[i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size]

            patch_bgr = image_bgr[i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size]
            # 计算梯度方向场
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            # 计算主导方向（加权平均）
            dominant_angle = np.average(angle, weights=mag) if np.sum(mag) > 1e-6 else 0

            # 旋转对齐主导方向
            M = cv2.getRotationMatrix2D((patch_size / 2, patch_size / 2), dominant_angle, 1)
            rotated = cv2.warpAffine(patch, M, (patch_size, patch_size))

            # 计算DCT频率特征
            dct = cv2.dct(rotated - np.mean(rotated))  # 20*20
            freq_weights = np.fromfunction(
                lambda u, v: np.sqrt((u / (patch_size - 1)) ** 2 + (v / (patch_size - 1)) ** 2),
                (patch_size, patch_size)  # enhance high-freq components
            )
            weighted_dct = np.abs(dct) * freq_weights
            # weighted_dct = torch.from_numpy(weighted_dct)
            f_magnitude[i, j] = np.sum(weighted_dct) / (patch_size ** 2)  # avg over patch_size
            f_orientation[i, j] = float(dominant_angle)

            if f_magnitude[i, j] > 0:
                # drawed_patch = draw_oriented_arrows_per_patch(patch_bgr, f_orientation[i, j], patch_size // 2,
                #                                               patch_size // 2, f_magnitude[i, j], None, True,
                #                                               None)  # blue
                drawed_patch = patch_bgr.copy()
                h, w = drawed_patch.shape[:2]
                center = (w // 2, h // 2)

                angle_rad = np.deg2rad(dominant_angle)
                length = int(min(h, w) * 0.3)
                # dx = int(length * np.cos(angle_rad));dy = int(length * np.sin(angle_rad))
                dx = int(length * np.cos(angle_rad));dy = int(-length * np.sin(angle_rad)) # (0,0) at left-up corner

                start_point = center
                end_point = (center[0] + dx, center[1] + dy)

                cv2.arrowedLine(drawed_patch, start_point, end_point, (255,0,0), thickness=1, tipLength=0.3)
                full_img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = drawed_patch
    cv2.imwrite(output_path, full_img)

# visualize_per_patch_freq('aa.png', patch_size=364)
visualize_per_patch_freq('bicycle_patch/input/_DSC8681_100.jpg', 'bicycle_patch/arrow_DSC8681_100png', blur=False, patch_size=20)
visualize_per_patch_freq('bicycle_patch/input/_DSC8682_100.jpg', 'bicycle_patch/arrow_DSC8682_100.png', blur=False, patch_size=20)
visualize_per_patch_freq('bicycle_patch/input/_DSC8685_100.jpg', 'bicycle_patch/arrow_DSC8685_100.png', blur=False, patch_size=20)