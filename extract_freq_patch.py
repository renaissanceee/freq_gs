import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.freq_utils import compute_oriented_frequencies, visualize_per_patch_freq
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import json
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import math

def save_results_json(final_frequencies, save_path="point_frequencies.json"):
    results = []
    for idx, (xyz, mag, dir) in final_frequencies.items():
        results.append({
            "voxel_xyz": xyz.tolist(),
            "voxel_mag": float(mag),
            "voxel_dir": dir.tolist()
        })
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"saved {save_path}")

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, final_iter):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # initial_setup
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    patch_size = 20
    viewpoint_stack = scene.getTrainCameras().copy()

    # adopt median for |freq|
    print("Update median for freq_mag...")
    voxel_xyz = gaussians.get_xyz.detach().cpu().numpy()
    voxel_dir = gaussians.get_direction.detach().cpu().numpy()

    point_frequencies = {i: [] for i in range(voxel_xyz.shape[0])}
    # scan all views
    for viewpoint_cam in tqdm(viewpoint_stack):# , desc="Processing cameras"
        gt_image = viewpoint_cam.original_image

        # before projection: R,t,K
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width
        # f_mag, f_orient = compute_oriented_frequencies(gt_image, patch_size)
        patch_size = 20
        output_dir = os.path.join(args.model_path, f"patch_{patch_size}", viewpoint_cam.image_name)
        visualize_per_patch_freq(output_dir, gt_image, patch_size)

    #     continue
    #     f_mag, f_orient = f_mag.numpy(), f_orient.numpy()
    #     R, t, K = viewpoint_cam.R, viewpoint_cam.T, viewpoint_cam.K
    #
    #     # 3. 处理每个3D点
    #     visible_pt = 0
    #     for pt_id in range(voxel_xyz.shape[0]):
    #         pt = voxel_xyz[pt_id,:]
    #         pt_cam = R @ pt + t
    #         if pt_cam[2] <= 0: continue  # 在相机后方
    #
    #         pt_2d = (K @ pt_cam) / pt_cam[2]
    #         x, y = pt_2d[:2].astype(int)
    #
    #         if 0 <= x < w and 0 <= y < h:  # fall into which patch
    #             patch_x = min(x // patch_size, f_mag.shape[1] - 1)
    #             patch_y = min(y // patch_size, f_mag.shape[0] - 1)
    #             distance = np.linalg.norm(pt_cam)  # sqrt(X²+Y²+Z²)
    #             freq_mag = f_mag[patch_y, patch_x] / (distance ** 0.8)  # or remove **0.8
    #             point_frequencies[pt_id].append(freq_mag)
    #     #         visible_pt += 1
    #     # print(viewpoint_cam.image_name)
    #     # print(visible_pt)
    #
    # # 计算最终频率特征
    # print("Aggregating frequencies...")
    # final_frequencies = {}
    # new_idx = 0
    # for i in range(voxel_xyz.shape[0]):
    #     if point_frequencies[i]: # if never visible, then []
    #         mags = np.array(point_frequencies[i])
    #         voxel_mag = np.median(mags)
    #         final_frequencies[i] = (
    #             voxel_xyz[i, :],  # position
    #             voxel_mag,  # 幅度中位数 e.g. 0~6.9
    #             voxel_dir[i, :])
    #         new_idx += 1
    # print(f'init voxels={voxel_xyz.shape[0]}')
    # print(f'non-empty voxels={new_idx}')
    #
    # # save_results_json(final_frequencies, os.path.join(args.model_path,"voxel_frequencies_voxsize5_lr0.05_100iter.json"))
    #
    # magg = [value[1] for value in final_frequencies.values()]
    # print(f'magg range...{max(magg), min(magg)}')
    #
    # xyzz = [value[0] for value in final_frequencies.values()]
    # xx = [value[0] for value in xyzz]
    # yy = [value[1] for value in xyzz]
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
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--final_iter", type=int, default=100)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.final_iter)

    # All done
    print("\nTraining complete.")
