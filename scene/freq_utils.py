import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import torchvision.transforms.functional as F
import torch
from scene.visualize_utils import read_json, draw_oriented_arrows, draw_mag, draw_oriented_arrows_per_patch

def visualize_per_patch_freq(output_dir, image, patch_size=20):
    image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    image = (image * 255).clip(0, 255).astype(np.uint8)  # = cv.imread
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    n_patches_y = h // patch_size
    n_patches_x = w // patch_size

    os.makedirs(output_dir, exist_ok=True)

    # f_magnitude = torch.zeros((n_patches_y, n_patches_x))
    # f_orientation = torch.zeros((n_patches_y, n_patches_x))
    f_magnitude = np.zeros((n_patches_y, n_patches_x))
    f_orientation = np.zeros((n_patches_y, n_patches_x))
    full_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n_patches_y): # edge px not involved
        for j in range(n_patches_x):
            patch = gray[i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size]
            
            patch_bgr = image_bgr[i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size]
            # # 计算梯度方向场
            # gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            # gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            # mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            #
            # # 计算主导方向（加权平均）
            # dominant_angle = np.average(angle, weights=mag) if np.sum(mag) > 1e-6 else 0
            #
            # # 旋转对齐主导方向
            # M = cv2.getRotationMatrix2D((patch_size / 2, patch_size / 2), dominant_angle, 1)
            # rotated = cv2.warpAffine(patch, M, (patch_size, patch_size))
            #
            # # 计算DCT频率特征
            # dct = cv2.dct(rotated - np.mean(rotated)) # 20*20
            # freq_weights = np.fromfunction(
            #     lambda u, v: np.sqrt((u / (patch_size - 1)) ** 2 + (v / (patch_size - 1)) ** 2),
            #     (patch_size, patch_size) # enhance high-freq components
            # )
            # weighted_dct = np.abs(dct) * freq_weights
            # # weighted_dct = torch.from_numpy(weighted_dct)
            # f_magnitude[i, j] = np.sum(weighted_dct) / (patch_size ** 2) # avg over patch_size
            # f_orientation[i, j] = float(dominant_angle)

            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            # 主导方向：使用加权向量平均
            if np.sum(mag) > 1e-6:
                angle_rad = np.deg2rad(angle)
                sin_sum = np.sum(np.sin(angle_rad) * mag)
                cos_sum = np.sum(np.cos(angle_rad) * mag)
                dominant_angle = (np.rad2deg(np.arctan2(sin_sum, cos_sum))) % 360
            else:
                dominant_angle = 0

            # 旋转 patch
            patch_size = patch.shape[0]
            M = cv2.getRotationMatrix2D((patch_size // 2, patch_size // 2), dominant_angle, 1)
            rotated = cv2.warpAffine(patch, M, (patch_size, patch_size))

            # 计算 DCT 频率
            rotated_f32 = rotated.astype(np.float32)
            rotated_f32 -= np.mean(rotated_f32)
            dct = cv2.dct(rotated_f32)

            freq_weights = np.fromfunction(
                lambda u, v: np.sqrt((u / (patch_size - 1)) ** 2 + (v / (patch_size - 1)) ** 2),
                (patch_size, patch_size), dtype=np.float32
            )
            weighted_dct = np.abs(dct) * freq_weights

            # 保存特征
            f_magnitude[i, j] = np.sum(weighted_dct) / (patch_size ** 2)
            f_orientation[i, j] = float(dominant_angle)

            if f_magnitude[i, j] >0:
                output_path = os.path.join(output_dir, f'{j}_{i}.png')
                drawed_patch = draw_oriented_arrows_per_patch(patch_bgr, f_orientation[i, j], patch_size//2, patch_size//2, f_magnitude[i, j], None, True, output_path)  # blue
                full_img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = drawed_patch
    cv2.imwrite(f'{output_dir}/full_image_with_arrows.png', full_img)


def compute_oriented_frequencies(image, patch_size=20):
    """计算带方向的频率特征"""
    # image = image.permute(1, 2, 0).cpu().numpy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    image = (image * 255).clip(0, 255).astype(np.uint8)  # = cv.imread
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    n_patches_y = h // patch_size
    n_patches_x = w // patch_size

    f_magnitude = torch.zeros((n_patches_y, n_patches_x))
    f_orientation = torch.zeros((n_patches_y, n_patches_x))

    for i in range(n_patches_y): # edge px not involved
        for j in range(n_patches_x):
            patch = gray[i * patch_size:(i + 1) * patch_size,
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
            dct = cv2.dct(rotated - np.mean(rotated)) # 20*20
            freq_weights = np.fromfunction(
                lambda u, v: np.sqrt((u / (patch_size - 1)) ** 2 + (v / (patch_size - 1)) ** 2),
                (patch_size, patch_size) # enhance high-freq components
            )
            weighted_dct = np.abs(dct) * freq_weights
            weighted_dct = torch.from_numpy(weighted_dct)
            f_magnitude[i, j] = torch.sum(weighted_dct) / (patch_size ** 2) # avg over patch_size
            f_orientation[i, j] = float(dominant_angle)# torch.tensor # torch.tensor(dominant_angle, dtype=torch.float32)

    return f_magnitude, f_orientation


def compute_oriented_frequencies_reload(image_path, patch_size=20):
    """计算带方向的频率特征"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    # image = (image * 255).clip(0, 255).astype(np.uint8)  # = cv.imread
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    n_patches_y = h // patch_size
    n_patches_x = w // patch_size

    f_magnitude = torch.zeros((n_patches_y, n_patches_x))
    f_orientation = torch.zeros((n_patches_y, n_patches_x))

    for i in range(n_patches_y): # edge px not involved
        for j in range(n_patches_x):
            patch = gray[i * patch_size:(i + 1) * patch_size,
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
            dct = cv2.dct(rotated - np.mean(rotated)) # 20*20
            freq_weights = np.fromfunction(
                lambda u, v: np.sqrt((u / (patch_size - 1)) ** 2 + (v / (patch_size - 1)) ** 2),
                (patch_size, patch_size) # enhance high-freq components
            )
            weighted_dct = np.abs(dct) * freq_weights
            weighted_dct = torch.from_numpy(weighted_dct)
            f_magnitude[i, j] = torch.sum(weighted_dct) / (patch_size ** 2) # avg over patch_size
            f_orientation[i, j] = float(dominant_angle)# torch.tensor # torch.tensor(dominant_angle, dtype=torch.float32)

    return f_magnitude, f_orientation
