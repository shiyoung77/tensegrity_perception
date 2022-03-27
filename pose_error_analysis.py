# Anaconda recommended
# python >= 3.6 required

import os
import importlib
import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import scipy.spatial.distance as distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    parser = ArgumentParser("pose evalutation")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--video", default="monday_roll15")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--num_endcaps", default=6, type=int)
    parser.add_argument("--mocap_scale", default=1000, type=int, help="scale of measured position (mm by default)")
    args = parser.parse_args()

    # read data config
    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    pose_folder = os.path.join(args.dataset, args.video, 'poses')
    positions = dict()
    for i in range(args.num_endcaps):
        positions[i] = np.load(os.path.join(pose_folder, f'{i}_pos.npy'))

    num_rods = len(data_cfg['color_to_rod'])
    N = positions[0].shape[0]  # number of estimated frames

    # load transformation from camera to motion capture
    cam_to_mocap_filepath = os.path.join(args.dataset, args.video, "cam_to_mocap.npy")
    if not os.path.exists(cam_to_mocap_filepath):
        print("[WARNING] Transformation not found. Start estimating... It may not be accurate.")
        os.system("python compute_T_from_cam_to_mocap.py")
    T = np.load(cam_to_mocap_filepath)
    R = T[:3, :3]
    t = T[:3, 3:4]

    ########################################################################################
    # compute rod pose error and endcap position error for each frame
    rod_error_data = defaultdict(list)
    endcap_error_data = defaultdict(list)

    for i in range(N):  # for each frame
        pts_est = np.zeros((args.num_endcaps, 3))
        for u in range(args.num_endcaps):
            pts_est[u] = positions[u][i]

        pts_mc = np.zeros((args.num_endcaps, 3))
        with open(os.path.join(args.dataset, args.video, 'data', f"{i + args.start_frame:04d}.json"), 'r') as f:
            info = json.load(f)
        for u in range(args.num_endcaps):
            pos = info['mocap'][str(u)]
            if pos is None:
                pts_mc[u] = [0, 0, 0]
            else:
                x = pos['x'] / args.mocap_scale  # (mm) -> (m)
                y = pos['y'] / args.mocap_scale
                z = pos['z'] / args.mocap_scale
                pts_mc[u] = [x, y, z]

        Q = pts_mc.T  # (3, num_endcaps)
        P = pts_est.T  # (3, num_endcaps)
        P_hat = R @ P + t  # (3, num_endcaps)

        # compute rod pose error
        for color, (u, v) in data_cfg['color_to_rod'].items():
            motion_capture_fail = np.allclose(Q[:, u], 0) or np.allclose(Q[:, v], 0)
            if not motion_capture_fail:
                rod_error_data['color'].append(color)
                rod_error_data['frame_id'].append(i)

                # translation error
                a = (P_hat[:, u] + P_hat[:, v]) / 2
                b = (Q[:, u] + Q[:, v]) / 2
                trans_err = distance.euclidean(a, b)
                rod_error_data['trans_err(m)'].append(trans_err)

                # rotation error
                a = P_hat[:, u] - P_hat[:, v]
                b = Q[:, u] - Q[:, v]
                rot_err = np.rad2deg(np.arccos(1 - distance.cosine(a, b)))
                rod_error_data['rot_err(deg)'].append(rot_err)

        # compute endcap position error
        for u in range(args.num_endcaps):
            motion_capture_fail = np.allclose(Q[:, u], 0)
            if not motion_capture_fail:
                endcap_error_data['frame_id'].append(i)
                endcap_error_data['error(m)'].append(distance.euclidean(Q[:, u], P_hat[:, u]))
                endcap_error_data['endcap_id'].append(u)

    rod_error_df = pd.DataFrame(data=rod_error_data)
    endcap_error_df = pd.DataFrame(data=endcap_error_data)

    ########################################################################################
    # rod pose error visualization
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(18, 10)
    axes[0, 0].grid(True)
    axes[0, 1].grid(True)
    sns.scatterplot(ax=axes[0, 0], x="frame_id", y="trans_err(m)", hue="color", data=rod_error_df)
    sns.scatterplot(ax=axes[0, 1], x="frame_id", y="rot_err(deg)", hue="color", data=rod_error_df)
    sns.boxplot(ax=axes[1, 0], x="color", y="trans_err(m)", data=rod_error_df)
    sns.boxplot(ax=axes[1, 1], x="color", y="rot_err(deg)", data=rod_error_df)
    plt.savefig(os.path.join(args.dataset, args.video, "rod_pose_error.png"))

    # rod pose error visualization
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(18, 10)
    axes[0].grid(True)
    sns.scatterplot(ax=axes[0], x="frame_id", y="error(m)", hue="endcap_id", data=endcap_error_df, palette="deep")
    sns.barplot(ax=axes[1], x="endcap_id", y="error(m)", data=endcap_error_df, palette="deep")
    plt.savefig(os.path.join(args.dataset, args.video, "endcap_position_error.png"))

    plt.show()
