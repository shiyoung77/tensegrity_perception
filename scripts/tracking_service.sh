#!/bin/bash

DATASET="dataset"
VIDEO="pebbles9"

METHOD="proposed"
START_FRAME=0
END_FRAME=10000  # exclusive

python tracking_service.py \
    --dataset $DATASET \
    --video $VIDEO \
    --rod_mesh_file "pcd/yale/struct_with_socks_new.ply" \
    --top_endcap_mesh_file "pcd/yale/end_cap_top_new.obj" \
    --bottom_endcap_mesh_file "pcd/yale/end_cap_bottom_new.obj" \
    --method $METHOD \
    --start_frame $START_FRAME \
    --end_frame $END_FRAME \
    --num_dummy_points 50 \
    --dummy_weights 0.5 \
    --render_scale 1 \
    --max_correspondence_distances 0.3 0.25 0.2 0.15 0.1 0.07 0.05 \
    --use_adaptive_weights \
    --optimize_every_n_iters 2 \
    --add_ground_constraints \
    --add_physical_constraints \
    --filter_observed_pts \
    --visualize \
    --save
