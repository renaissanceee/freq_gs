import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.freq_utils import compute_oriented_frequencies, compute_sobel_grad, visualize_per_patch_freq, visualize_per_pix_sobel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import json
from scipy.spatial.distance import cosine
import math

# lift f_orient to 3d: img-plane->cam->world
def lift_orient_2d_to_3d(f_orient_cur, R, K):
    d_image = np.array([np.cos(f_orient_cur), np.sin(f_orient_cur), 1])
    d_cam = np.linalg.inv(K) @ d_image  # K^{-1} * d_image
    d_cam_normalized = d_cam / np.linalg.norm(d_cam)  #
    d_world = R.T @ d_cam_normalized  # R^T * d_cam
    d_world = d_world / np.linalg.norm(d_world)
    return d_world

def save_results_json(final_results, save_path="point_frequencies.json"):
    results = []
    for idx, (xyz, mag, dir) in final_results.items():
        xyz = xyz.tolist()
        dir = dir.tolist() if dir is not None else None
        results.append({
            "voxel_xyz": xyz,
            "voxel_mag": float(mag),
            "voxel_dir": dir
        })
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"saved {save_path}")

def freq_per_patch(dataset, opt, pipe):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    patch_size = 20
    viewpoint_stack = scene.getTrainCameras().copy()

    # adopt median for |freq|
    print("Update median for freq ...")
    voxel_xyz = gaussians.get_xyz.detach().cpu().numpy()
    voxel_dir = gaussians.get_direction.detach().cpu().numpy()

    point_frequencies = {i: [] for i in range(voxel_xyz.shape[0])}
    patch_size = 20
    # scan all views
    for viewpoint_cam in tqdm(viewpoint_stack):# , desc="Processing cameras"
        gt_image = viewpoint_cam.original_image

        # visualize 2d
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width
        output_dir = os.path.join(args.model_path, f"patch_{patch_size}", viewpoint_cam.image_name)
        visualize_per_patch_freq(output_dir, gt_image, patch_size)

        # before projection: R,t,K
        f_mag, f_orient = compute_oriented_frequencies(gt_image, patch_size)
        f_mag, f_orient = f_mag.numpy(), f_orient.numpy()
        R, t, K = viewpoint_cam.R, viewpoint_cam.T, viewpoint_cam.K

        # 处理每个3D点
        visible_pt = 0
        for pt_id in range(voxel_xyz.shape[0]):
            pt = voxel_xyz[pt_id,:]
            pt_cam = R @ pt + t
            if pt_cam[2] <= 0: continue  # 在相机后方

            pt_2d = (K @ pt_cam) / pt_cam[2]
            x, y = pt_2d[:2].astype(int)

            if 0 <= x < w and 0 <= y < h:  # fall into which patch
                patch_x = min(x // patch_size, f_mag.shape[1] - 1)
                patch_y = min(y // patch_size, f_mag.shape[0] - 1)
                distance = np.linalg.norm(pt_cam)  # sqrt(X²+Y²+Z²)
                freq_mag = f_mag[patch_y, patch_x] / (distance ** 0.8)  # or remove **0.8
                point_frequencies[pt_id].append(freq_mag)
        #         visible_pt += 1
        # print(viewpoint_cam.image_name)
        # print(visible_pt)

    # 计算最终频率特征
    print("Aggregating frequencies...")
    final_frequencies = {}
    new_idx = 0
    for i in range(voxel_xyz.shape[0]):
        if point_frequencies[i]: # if never visible, then []
            mags = np.array(point_frequencies[i])
            voxel_mag = np.median(mags)
            final_frequencies[i] = (
                voxel_xyz[i, :],  # position
                voxel_mag,  # 幅度中位数 e.g. 0~6.9
                voxel_dir[i, :])
            new_idx += 1
    print(f'init voxels={voxel_xyz.shape[0]}')
    print(f'non-empty voxels={new_idx}')

    # save_results_json(final_frequencies, os.path.join(args.model_path,"voxel_frequencies_voxsize5_lr0.05_100iter.json"))

    magg = [value[1] for value in final_frequencies.values()]
    print(f'magg range...{max(magg), min(magg)}')

    xyzz = [value[0] for value in final_frequencies.values()]
    xx = [value[0] for value in xyzz]
    yy = [value[1] for value in xyzz]
    # zz = [value[2] for value in xyzz]
    # print(f'x range...{min(xx), max(xx)}')
    # print(f'y range...{min(yy), max(yy)}')
    # print(f'z range...{min(zz), max(zz)}')


def sobel_per_pix(dataset, opt, pipe):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    viewpoint_stack = scene.getTrainCameras().copy()

    # adopt median for |freq|
    print("Update median for gards ...")
    voxel_xyz = gaussians.get_xyz.detach().cpu().numpy()
    voxel_dir = gaussians.get_direction.detach().cpu().numpy()

    point_mag = {i: [] for i in range(voxel_xyz.shape[0])}
    point_dir = {i: [] for i in range(voxel_xyz.shape[0])}
    patch_size = 5

    output_dir_mag = os.path.join(args.model_path, 'sobel', 'mag')
    output_dir_dir = os.path.join(args.model_path, 'sobel', 'dir')
    os.makedirs(output_dir_mag, exist_ok=True)
    os.makedirs(output_dir_dir, exist_ok=True)
    # scan all views
    for viewpoint_cam in tqdm(viewpoint_stack):  # , desc="Processing cameras"
        gt_image = viewpoint_cam.original_image

        # visualize 2d
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width
        visualize_per_pix_sobel(output_dir_mag, output_dir_dir, viewpoint_cam.image_name, gt_image, threshold=50, down=None) # down=5

        # before projection: R,t,K
        # f_mag, f_orient = compute_oriented_frequencies(gt_image, patch_size)
        f_mag, f_orient = compute_sobel_grad(gt_image, threshold=50) # np.array
        R, t, K = viewpoint_cam.R, viewpoint_cam.T, viewpoint_cam.K

        # 处理每个3D点
        visible_pt = 0
        for pt_id in range(voxel_xyz.shape[0]):
            pt = voxel_xyz[pt_id, :]
            pt_cam = R @ pt + t
            if pt_cam[2] <= 0: continue  # 在相机后方

            pt_2d = (K @ pt_cam) / pt_cam[2]
            # x, y = pt_2d[:2].astype(int)
            x, y = int(round(pt_2d[0])), int(round(pt_2d[1]))

            if 0 <= x < w and 0 <= y < h:  # fall into which patch
                distance = np.linalg.norm(pt_cam)  # sqrt(X²+Y²+Z²)
                ## 1) directly use nearest
                # f_mag_cur, f_orient_cur = f_mag[y, x], f_orient[y, x]

                ## 2) max neighbor
                # local patch_size
                y1, y2 = max(0, y - patch_size),min(h, y + patch_size + 1)
                x1, x2 = max(0, x - patch_size), min(w, x + patch_size + 1)
                patch = f_mag[y1:y2, x1:x2]
                dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
                max_y, max_x = y1 + dy, x1 + dx
                f_mag_cur, f_orient_cur = f_mag[max_y, max_x], f_orient[max_y, max_x]
                
                if f_orient_cur is not None:
                    d_world = lift_orient_2d_to_3d(f_orient_cur, R, K)
                    f_mag_dist = f_mag_cur / (distance ** 0.8)  # or remove **0.8
                    point_mag[pt_id].append(f_mag_dist)
                    point_dir[pt_id].append(d_world)

                # visible_pt += 1
        # print(viewpoint_cam.image_name)
        # print(visible_pt)


    # 计算最终频率特征
    print("Aggregating grads...")
    final_grads = {}
    visible_pt = 0
    noise_pt = 0
    cos_threhold = 0.7 # 45°:√2/2=0.707   60°:0.5
    for i in range(voxel_xyz.shape[0]):
        if point_mag[i]:  # if never visible, then []
            # mag
            mags = np.array(point_mag[i])
            median_mag = np.median(mags)
            median_index = np.abs(mags - np.median(median_mag)).argmin()
            max_mag = np.max(mags)
            max_index = np.where(mags == max_mag)[0][0]
            min_mag = np.min(mags)
            min_index = np.where(mags == min_mag)[0][0]
            # dir
            dirs = np.array(point_dir[i])
            median_dir, max_dir, min_dir = dirs[median_index,:], dirs[max_index,:], dirs[min_index,:]
            # check variance
            # cos_sim = np.abs(cosine(median_dir, max_dir))
            cos_sim = np.abs(cosine(median_dir, min_dir))
            if cos_sim<cos_threhold:
                median_dir = None
                noise_pt+=1

            final_grads[i] = (
                voxel_xyz[i, :],  # position
                median_mag,  # 幅度中位数 e.g. 0~6.9
                median_dir)  # 朝向中位数
            visible_pt += 1
    print(f'init voxels={voxel_xyz.shape[0]}') # 54275
    print(f'non-empty voxels={visible_pt}') # 53206
    print(f'noise voxels={noise_pt}') # 2966

    save_results_json(final_grads, os.path.join(args.model_path,f"voxel_grads_colmap_cos{cos_threhold}.json"))

    magg = [value[1] for value in final_grads.values()]
    print(f'magg range...{max(magg), min(magg)}')

    xyzz = [value[0] for value in final_grads.values()]
    xx = [value[0] for value in xyzz]
    yy = [value[1] for value in xyzz]
    # zz = [value[2] for value in xyzz]
    # print(f'x range...{min(xx), max(xx)}')
    # print(f'y range...{min(yy), max(yy)}')
    # print(f'z range...{min(zz), max(zz)}')

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)
    
    # freq_per_patch(lp.extract(args), op.extract(args), pp.extract(args)) # patch-wise

    sobel_per_pix(lp.extract(args), op.extract(args), pp.extract(args)) # pix-wise

