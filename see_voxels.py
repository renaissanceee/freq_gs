import os
from random import randint
import sys
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from scene.visualize_utils import read_json, draw_oriented_arrows, draw_mag

def visualize_freq(dataset, opt, pipe, json_file, output_dir, top_percent):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    viewpoint_stack = scene.getTrainCameras().copy()
    print(f'load freq from ... {json_file}')
    positions_base, freqs_base, freqs_norm_base, orientations_base = read_json(json_file)
    os.makedirs(output_dir, exist_ok=True)
    # -----------------------
    ### 3d-grids ####
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(vmin=freqs_base.min(), vmax=freqs_base.max())
    colors = cm.viridis(norm(freqs_base))

    sc = ax.scatter(positions_base[:, 0], positions_base[:, 1], positions_base[:, 2],
                    c=colors, s=10, marker='o')

    mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array(freqs_base)
    fig.colorbar(mappable, ax=ax, label='voxel_mag')
    ax.set_title("Voxel Cloud Colored by Magnitude (viridis)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    # ax.view_init(elev=30, azim=45) # another view
    plt.savefig(os.path.join(output_dir, "3d_grids_voxel_tensor2gray.png"), dpi=300)# "3d_grids_colmap.png"
    plt.close()
    # -----------------------
    for viewpoint_cam in tqdm(viewpoint_stack):# , desc="Processing cameras"
        filename = viewpoint_cam.image_name
        gt_image = viewpoint_cam.original_image.cpu()
        gt_image = gt_image.permute(1, 2, 0).numpy()[:, :, ::-1]  # [H, W, C]
        gt_image = (gt_image * 255).clip(0, 255).astype(np.uint8) # = cv.imread
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width

        R, t, K = viewpoint_cam.R, viewpoint_cam.T.reshape(3, 1), viewpoint_cam.K

        # 投影
        points_cam = (R @ positions_base.T + t).T
        in_front = points_cam[:, 2] > 0
        points_cam = points_cam[in_front]
        freqs_norm = freqs_norm_base[in_front]
        orientations = orientations_base[in_front]
        # proj position and direction
        # position
        points_proj = (K @ points_cam.T).T
        points_proj = (points_proj[:, :2].T / points_proj[:, 2]).T
        # pt_cam = R @ positions_base + t;pt_2d = (K @ pt_cam) / pt_cam[2]

        # direction
        orientations_proj = (R @ orientations.T).T  # [N, 3]
        orientations_proj = orientations_proj[:, :2]
        orientations_proj = orientations_proj / np.linalg.norm(orientations_proj, axis=1, keepdims=True)

        x, y = points_proj[:, 0], points_proj[:, 1]
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y, orientations_proj, freqs_norm = x[valid], y[valid], orientations_proj[valid], freqs_norm[valid]
        colors = plt.cm.viridis(freqs_norm)[:, :3] * 255
        print(f"visible points = {x.shape[0]}")
        colors_bgr = (colors[:, ::-1]).astype(np.uint8)

        # _____________________________________
        # 1) orientation
        # top x%
        freq_threshold = np.percentile(freqs_norm, top_percent)
        mask_top = freqs_norm > freq_threshold
        x_top, y_top = x[mask_top], y[mask_top]
        orientations_proj_top = orientations_proj[mask_top]
        freqs_top = freqs_norm[mask_top]
        colors_top = colors_bgr[mask_top]
        output_path=os.path.join(output_dir, f"oriented_freq_{filename}_top{top_percent}percent.png")
        draw_oriented_arrows(gt_image, orientations_proj_top, x_top, y_top, freqs_top, None, True, output_path) # blue
        draw_oriented_arrows(gt_image, orientations_proj_top, x_top, y_top, freqs_top, colors_top, False, output_path) # viridis
        # _____________________________________
        # 2) magnitude
        ## w/o color bar
        output_path = os.path.join(output_dir, f"mag_freq_{filename}.png")
        draw_mag(gt_image, x, y, h, w, freqs_norm, output_path)
        draw_mag(gt_image, x, y, h, w, freqs_norm, output_path, bg=False)
        # break



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--json', type=str, default="voxel_frequencies_voxel.json")
    parser.add_argument('--top', type=int, default=80)
    # args = parser.parse_args(sys.argv[1:])
    args = parser.parse_args()

    print("Visualize ... " + args.model_path)
    json_file = os.path.join(args.model_path, args.json)
    output_dir = os.path.splitext(args.json)[0].replace('voxel_frequencies', 'freq_grids')  # "freq_grids_voxel, "freq_grids_colmap"
    output_dir = os.path.join(args.model_path, output_dir)
    visualize_freq(lp.extract(args), op.extract(args), pp.extract(args), json_file, output_dir, args.top)
