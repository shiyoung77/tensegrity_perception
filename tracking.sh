#!/bin/bash

DATASET="dataset"

# VIDEO_LIST=({0001..0016})
# VIDEO_LIST=($(ls $DATASET))
# VIDEO_LIST=("pebbles9")
VIDEO_LIST=("evenlowercamera_3")
# VIDEO_LIST=($(ls $DATASET | grep -E "R2S2Rcrawling"))

METHOD="proposed"
START_FRAME=0
END_FRAME=10000  # exclusive

echo ${VIDEO_LIST[@]}
echo "# videos: ${#VIDEO_LIST[@]}"

for VIDEO in ${VIDEO_LIST[@]}; do
    if [[ $VIDEO == "" ]]
    then
        continue
    fi
    echo $VIDEO

    # PYOPENGL_PLATFORM="osmesa" python tracking.py \
    python tracking.py \
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
        --max_correspondence_distances 0.3 0.25 0.2 0.15 0.1 0.07 0.05 0.05 0.05 \
        --use_adaptive_weights \
        --optimize_every_n_iters 0 \
        --add_ground_constraints \
        --add_physical_constraints \
        --filter_observed_pts \
        --visualize \
        --save

    ffmpeg -r 10 -i "$DATASET/$VIDEO/estimation_vis-${METHOD}/%04d.jpg" \
        -start_number $START_FRAME \
        -vframes $(expr $END_FRAME - $START_FRAME) \
        "$DATASET/$VIDEO/estimation.mp4"

    # ffmpeg -i $DATASET/$VIDEO/estimation.mp4 \
    #     -t 20 \
    #     -vf "fps=10,scale=960:240:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    #     -loop 0 \
    #     $DATASET/$VIDEO/$VIDEO.gif

    # mkdir -p ~/dataset/dusk_results
    # cp -r $DATASET/$VIDEO/poses-proposed $DATASET/$VIDEO/estimation.mp4 $DATASET/$VIDEO/config.json --parents ~/dataset/dusk_results

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
