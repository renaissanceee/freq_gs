import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.freq_utils import compute_oriented_frequencies,compute_sobel_grad
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

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, final_iter, json_file, colmap):
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, colmap=colmap)
    gaussians.training_setup_voxel(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # initial_setup
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    patch_size = 20
    viewpoint_stack = scene.getTrainCameras().copy()
    # viewpoint_stack = viewpoint_stack[:2]

    # ----------------------------------------------------
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()
        voxel_xyz = gaussians.get_xyz
        voxel_dir = gaussians.get_direction
        voxel_confidence = gaussians.get_confidence

        # Pick a random Camera
        viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack)-1)]

        gt_image = viewpoint_cam.original_image.cuda()

        # before projection: R,t,K
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width
        f_mag, f_orient = compute_sobel_grad(gt_image)
        f_mag, f_orient = torch.from_numpy(f_mag).cuda(), torch.from_numpy(f_orient).cuda()
        R, t, K = torch.from_numpy(viewpoint_cam.R).float().cuda(), torch.from_numpy(
            viewpoint_cam.T).float().cuda(), torch.from_numpy(viewpoint_cam.K).float().cuda()

        pt_cam = (R @ voxel_xyz.T).T + t  # [N, 3]
        distances = torch.norm(pt_cam, p=2, dim=1) # decay?
        valid_mask = pt_cam[:, 2] > 0  # [N]
        pt_cam = pt_cam[valid_mask]  # [M, 3]
        distances = distances[valid_mask]
        voxel_dir = voxel_dir[valid_mask]  # [M, 3]
        voxel_confidence = voxel_confidence[valid_mask]
        pt_2d = (K @ pt_cam.T).T  # [M, 3]
        pt_2d = pt_2d / pt_2d[:, 2:]  # [M, 3]
        xy = pt_2d[:, :2].to(torch.int64)  # [M, 2] (x, y)
        x, y = xy[:, 0], xy[:, 1]
        boundary_mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)  # [M]
        x, y = x[boundary_mask], y[boundary_mask]  # [K, 2], K <= M
        distances = distances[boundary_mask]
        dir_2d = (R @ voxel_dir.T).T[boundary_mask]  # [K, 3]
        dir_2d = dir_2d[:, :2]  # [K, 2]
        dir_2d = dir_2d / (torch.norm(dir_2d, p=2, dim=1, keepdim=True) + 1e-8)  # 归一化
        conf_2d = voxel_confidence[boundary_mask]
        # import pdb;pdb.set_trace()
        # conf_2d_decay = conf_2d / (distances ** 0.8) # decay
        conf_2d_decay = conf_2d.squeeze(1)*torch.exp(-0.1 * distances)
        # import pdb;pdb.set_trace()

        # init an img-plane
        splat_dir = torch.zeros((h, w, 2), device=xy.device)  # [H, W, 2]
        splat_conf = torch.zeros((h, w), device=xy.device)  # [H, W]
        splat_count = torch.zeros((h, w), device=xy.device)  # [H, W]

        linear_indices = y * w + x
        # import pdb;pdb.set_trace()
        splat_dir.view(-1, 2).index_add_(0, linear_indices, dir_2d * conf_2d_decay.unsqueeze(1))
        splat_conf.view(-1).index_add_(0, linear_indices, conf_2d_decay)
        # splat_count.view(-1).index_add_(
        #     0, linear_indices, torch.ones_like(conf_2d_decay)
        # )
        valid_mask = splat_conf > 0
        splat_dir[valid_mask] = splat_dir[valid_mask] / splat_conf[valid_mask][:, None] # ?

        mask = (f_mag > 0) & (torch.norm(splat_dir, p=2, dim=2) > 0)  # gt!=0, and accumulated!=0
        splat_dir, f_orient = splat_dir[mask], f_orient[mask]

        gt_dir = torch.stack([torch.cos(f_orient), torch.sin(f_orient)], dim=-1).cuda() # [N,2]
        gt_dir = gt_dir / (torch.norm(gt_dir, p=2) + 1e-8)  # normaliz.

        # loss
        # import pdb;pdb.set_trace()
        cos_sim = torch.nn.functional.cosine_similarity(splat_dir, gt_dir, dim=1)
        loss = 1 - torch.abs(cos_sim).mean()

        loss.backward()
        if iteration % 1000 == 0:
            print(f'loss={loss}')
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            # # Optimizer step
            if iteration < final_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    # ----------------------------------------------------
    # save_json
    print("logging visible points...")
    voxel_xyz = gaussians.get_xyz
    voxel_dir = gaussians.get_direction
    voxel_confidence = gaussians.get_confidence
    final_results = {}
    visible_pt = 0
    for i in range(voxel_xyz.shape[0]):
        if voxel_confidence[i]>0.1:
            final_results[i] = (
                voxel_xyz[i, :],  # position
                voxel_confidence[i],
                voxel_dir[i, :])
            visible_pt += 1
    print(f'saved points={visible_pt}')
    save_results_json(final_results, json_file)

    # print(f'init voxels={voxel_xyz.shape[0]}')
    # print(f'non-empty voxels={new_idx}')

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

    # ----------------------------------------------------



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
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument('--colmap', action='store_true', help='Use .ply or uniform_grid')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    json_file = os.path.join(args.model_path, args.json)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.final_iter, json_file, args.colmap)

    # All done
    print("\nTraining complete.")
