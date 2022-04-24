#!/bin/bash

DATASET="dataset"
VIDEO_ID="shiyang6"

python tracking.py \
    --dataset $DATASET \
    --video_id $VIDEO_ID \
    --rod_mesh_file "pcd/yale/struct_with_socks_new.ply" \
    --top_endcap_mesh_file "pcd/yale/end_cap_top_new.obj" \
    --bottom_endcap_mesh_file "pcd/yale/end_cap_bottom_new.obj" \
    --first_frame_id 0 \
    --max_correspondence_distances 0.3 0.15 0.1 0.06 0.03 \
    --visualize \
    --add_fake_pts \
    --filter_observed_pts \
    --add_constrained_optimization \
    --add_physical_constraints