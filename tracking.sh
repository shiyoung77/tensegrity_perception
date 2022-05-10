#!/bin/bash

DATASET="dataset"
VIDEO_ID="0016"
START_FRAME=0
END_FRAME=1000  # exclusive

python tracking.py \
    --dataset $DATASET \
    --video_id $VIDEO_ID \
    --rod_mesh_file "pcd/yale/struct_with_socks_new.ply" \
    --top_endcap_mesh_file "pcd/yale/end_cap_top_new.obj" \
    --bottom_endcap_mesh_file "pcd/yale/end_cap_bottom_new.obj" \
    --start_frame $START_FRAME \
    --end_frame $END_FRAME \
    --render_scale 2 \
    --max_correspondence_distances 0.3 0.25 0.2 0.15 0.1 0.06 0.03 \
    --add_dummy_points \
    --num_dummy_points 50 \
    --dummy_weights 0.5 \
    --filter_observed_pts \
    --add_constrained_optimization \
    --add_ground_constraints \
    --add_physical_constraints \
    --visualize \

ffmpeg -r 10 -i "$DATASET/$VIDEO_ID/estimation_vis/%04d.png" \
    -start_number $START_FRAME \
    -vframes $(expr $END_FRAME - $START_FRAME) \
    "$DATASET/$VIDEO_ID/estimation.mp4"

# python compute_T_from_cam_to_mocap_manual.py \
#     --dataset $DATASET \
#     --video $VIDEO_ID \
#     --start_frame 0 \
#     --mocap_scale 1000

python compute_T_from_cam_to_mocap.py \
    --dataset $DATASET \
    --video $VIDEO_ID \
    --start_frame $START_FRAME \
    --end_frame $END_FRAME \
    --num_endcaps 6 \
    --mocap_scale 1000

python pose_error_analysis.py \
    --dataset $DATASET \
    --video $VIDEO_ID \
    --start_frame $START_FRAME \
    --end_frame $END_FRAME \
    --num_endcaps 6 \
    --mocap_scale 1000

python cable_length_error_analysis.py \
    --dataset $DATASET \
    --video $VIDEO_ID \
    --pose_folder "poses" \
    --start_frame $START_FRAME \
    --end_frame $END_FRAME \
    --num_endcaps 6 \
    --mocap_scale 1000

python compute_mocap_stats.py \
    --dataset $DATASET \
    --video $VIDEO_ID \
    --start_frame $START_FRAME \
