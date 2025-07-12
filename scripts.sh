python train_direction_0704.py --eval -r 8 -s ../dataset/360v2/bicycle -m cc --json voxel_frequencies_colmap.json --colmap
python see_voxels.py --eval -r 8 -s ../dataset/360v2/bicycle -m cc --json voxel_frequencies_colmap.json --min_conf 0.7

python train_direction_weighted.py --eval -r 8 -s ../dataset/360v2/bicycle -m cc --json weighted_sobel_colmap_10k.json --final_iter 10000
# gaussian_model.py里面改init_from_pcd
python extract_freq_or_grad.py --eval -r 8 -s ../dataset/360v2/bicycle -m cc

python render.py --eval -r 4 -s ../dataset/360v2/bicycle -m benchmark_bicycle_x4 --kernel_size 0.3