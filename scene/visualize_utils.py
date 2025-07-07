import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path

def read_json(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    positions = np.array([d["voxel_xyz"] for d in data])
    magnitudes = np.array([d["voxel_mag"] for d in data])
    freqs_norm = magnitudes / (magnitudes.max() + 1e-6) # normaliz.
    orientations = np.array([d["voxel_dir"] for d in data])
    return positions, magnitudes, freqs_norm, orientations

def draw_oriented_arrows_per_patch(gt_image, angle, x_coords, y_coords, freqs, colors, single_color, output_path):
    line_length = 10
    vis_img = gt_image.copy()
    angle = np.deg2rad(angle)
    start_point = (int(x_coords), int(y_coords))
    length = line_length # equal-len   # length = max(4 * line_length * freqs, line_length)
    # dx, dy = length * np.cos(angle), length * np.sin(angle)
    dx,dy = length * np.cos(angle), -length * np.sin(angle)
    end_point = (int(round(x_coords + dx)), int(round(y_coords + dy)))
    if single_color:
        cv2.arrowedLine(vis_img, start_point, end_point, (255, 0, 0), thickness=1, tipLength=0.2)  # bgr: red(0,0,255), blue(255,0,0)
    else:
        color_bgr = tuple(int(c) for c in colors)
        cv2.arrowedLine(vis_img, start_point, end_point, color_bgr, thickness=1, tipLength=0.2)

    output_path = output_path.replace('.', '_blue.') if single_color else output_path
    cv2.imwrite(output_path, vis_img)
    return vis_img

def draw_oriented_arrows(gt_image, orientations, x_coords, y_coords, freqs, colors, single_color, output_path):
    line_length = 10
    vis_img = gt_image.copy()
    for i in range(0, len(orientations)):
        angle = np.arctan2(orientations[i, 1], orientations[i, 0])  # [N,]
        start_point = (int(x_coords[i]), int(y_coords[i]))
        length = max(4 * line_length * freqs[i], line_length)
        # dx, dy = length * np.cos(angle), length * np.sin(angle)
        dx,dy = length * np.cos(angle), -length * np.sin(angle)
        end_point = (int(round(x_coords[i] + dx)), int(round(y_coords[i] + dy)))
        if single_color:
            cv2.arrowedLine(vis_img, start_point, end_point, (255, 0, 0), thickness=1, tipLength=0.2)  # bgr: red(0,0,255), blue(255,0,0)
        else:
            color_bgr = tuple(int(c) for c in colors[i])
            cv2.arrowedLine(vis_img, start_point, end_point, color_bgr, thickness=1, tipLength=0.2)

    output_path = output_path.replace('.', '_blue.') if single_color else output_path
    cv2.imwrite(output_path, vis_img)


def draw_mag(gt_image, x_coords, y_coords, h, w, freqs_norm, output_path, bg=True):
    vis_img = gt_image.copy()
    plt.figure(figsize=(w / 100, h / 100), dpi=100)
    plt.scatter(x_coords, y_coords, c=freqs_norm, cmap='viridis', s=1, alpha=0.4)
    plt.axis('off')
    plt.tight_layout(pad=0)
    if bg:
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), alpha=0.6)
    else:
        plt.gca().invert_yaxis()
        output_path = output_path.replace('.', '_scatter.')

    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
