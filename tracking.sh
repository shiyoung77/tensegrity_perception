#!/bin/bash

DATASET="dataset"
# VIDEO_LIST=({0001..0016})
# VIDEO_LIST=(15deg_2)
# VIDEO_LIST=(15deg_2 15deg_4 15deg_6 15deg_7 15deg_10 15deg_12)
# VIDEO_LIST=(10deg 10deg_{2..6})
VIDEO_LIST=($(ls $DATASET | grep -E "20deg"))
METHOD="proposed"
START_FRAME=0
END_FRAME=1000  # exclusive

echo "# videos: ${#VIDEO_LIST[@]}"

for VIDEO in ${VIDEO_LIST[@]}; do
    echo $VIDEO
    # if [[ $VIDEO != "25wide_44" ]]
    # then
    #     continue
    # fi

    python tracking.py \
        --dataset $DATASET \
        --video $VIDEO \
        --rod_mesh_file "pcd/yale/struct_with_socks_new.ply" \
        --top_endcap_mesh_file "pcd/yale/end_cap_top_new.obj" \
        --bottom_endcap_mesh_file "pcd/yale/end_cap_bottom_new.obj" \
        --method $METHOD \
        --start_frame $START_FRAME \
        --end_frame $END_FRAME \
        --add_dummy_points \
        --num_dummy_points 50 \
        --dummy_weights 0.5 \
        --render_scale 1 \
        --max_correspondence_distances 0.15 0.1 0.1 0.06 0.05 0.04 0.03 \
        --add_constrained_optimization \
        --add_ground_constraints \
        --add_physical_constraints \
        --filter_observed_pts \
        --visualize \
        --save

    ffmpeg -r 30 -i "$DATASET/$VIDEO/estimation_vis-${METHOD}/%04d.jpg" \
        -start_number $START_FRAME \
        -vframes $(expr $END_FRAME - $START_FRAME) \
        "$DATASET/$VIDEO/estimation.mp4"

    ffmpeg -i $DATASET/$VIDEO/estimation.mp4 \
        -t 20 \
        -vf "fps=10,scale=960:240:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
        -loop 0 \
        $DATASET/$VIDEO/$VIDEO.gif

    cp -r $DATASET/$VIDEO/poses-proposed $DATASET/$VIDEO/estimation.mp4 $DATASET/$VIDEO/$VIDEO.gif $DATASET/$VIDEO/config.json --parents /mnt/evo/dataset/new_results
    # break
done


# python compute_T_from_cam_to_mocap_manual.py \
#     --dataset $DATASET \
#     --video $VIDEO \
#     --start_frame 0 \
#     --mocap_scale 1000

# python compute_T_from_cam_to_mocap.py \
#     --dataset $DATASET \
#     --video $VIDEO \
#     --start_frame $START_FRAME \
#     --end_frame $END_FRAME \
#     --num_endcaps 6 \
#     --mocap_scale 1000

# python pose_error_analysis.py \
#     --dataset $DATASET \
#     --video $VIDEO \
#     --start_frame $START_FRAME \
#     --end_frame $END_FRAME \
#     --num_endcaps 6 \
#     --mocap_scale 1000

# python cable_length_error_analysis.py \
#     --dataset $DATASET \
#     --video $VIDEO \
#     --pose_folder "poses" \
#     --start_frame $START_FRAME \
#     --end_frame $END_FRAME \
#     --num_endcaps 6 \
#     --mocap_scale 1000

# python compute_mocap_stats.py \
#     --dataset $DATASET \
#     --video $VIDEO \
#     --start_frame $START_FRAME \
