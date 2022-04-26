#!/bin/bash

DATASET="dataset"
VIDEO_ID="monday_roll15"

python tracking.py \
    --dataset $DATASET \
    --video_id $VIDEO_ID \
    --rod_mesh_file "pcd/yale/struct_with_socks_new.ply" \
    --top_endcap_mesh_file "pcd/yale/end_cap_top_new.obj" \
    --bottom_endcap_mesh_file "pcd/yale/end_cap_bottom_new.obj" \
    --start_frame 0 \
    --max_correspondence_distances 0.3 0.25 0.2 0.15 0.1 0.07 0.04 0.02 \
    --add_fake_pts \
    --filter_observed_pts \
    --add_constrained_optimization \
    --add_physical_constraints \
    --visualize \

# python compute_T_from_cam_to_mocap_manual.py \
#     --dataset $DATASET \
#     --video $VIDEO_ID \
#     --start_frame 0 \
#     --mocap_scale 1000

python compute_T_from_cam_to_mocap.py \
    --dataset $DATASET \
    --video $VIDEO_ID \
    --start_frame 0 \
    --num_endcaps 6 \
    --mocap_scale 1000

python pose_error_analysis.py \
    --dataset $DATASET \
    --video $VIDEO_ID \
    --start_frame 0 \
    --num_endcaps 6 \
    --mocap_scale 1000

python cable_length_error_analysis.py \
    --dataset $DATASET \
    --video $VIDEO_ID \
    --pose_folder "poses" \
    --start_frame 0 \
    --num_endcaps 6 \
    --mocap_scale 1000

python compute_mocap_stats.py \
    --dataset $DATASET \
    --video $VIDEO_ID \
    --start_frame 0 \
