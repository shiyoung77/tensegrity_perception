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


def eval_single_video(args, video, rod_error_data, endcap_error_data):
    print(f'{args.dataset}.{video}.config')
    data_cfg_module = importlib.import_module(f'{args.dataset}.{video}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    pose_folder = os.path.join(args.dataset, video, 'poses')
    positions = dict()
    for i in range(args.num_endcaps):
        positions[i] = np.load(os.path.join(pose_folder, f'{i}_pos.npy'))

    N = min(positions[0].shape[0], args.end_frame)

    # load transformation from camera to motion capture
    cam_to_mocap_filepath = os.path.join(args.dataset, video, "cam_to_mocap.npy")
    if not os.path.exists(cam_to_mocap_filepath):
        print("[WARNING] Transformation not found. Start estimating... It may not be accurate.")
        os.system(f"python compute_T_from_cam_to_mocap.py -v {video}")
    T = np.load(cam_to_mocap_filepath)
    R = T[:3, :3]
    t = T[:3, 3:4]

    for i in range(N):  # for each frame
        pts_est = np.zeros((args.num_endcaps, 3))
        for u in range(args.num_endcaps):
            pts_est[u] = positions[u][i]

        pts_mc = np.zeros((args.num_endcaps, 3))
        with open(os.path.join(args.dataset, video, 'data', f"{i + args.start_frame:04d}.json"), 'r') as f:
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
                rod_error_data['trans_err (cm)'].append(trans_err * 100)

                # rotation error
                a = P_hat[:, u] - P_hat[:, v]
                b = Q[:, u] - Q[:, v]
                rot_err = np.rad2deg(np.arccos(1 - distance.cosine(a, b)))
                rod_error_data['rot_err (deg)'].append(rot_err)

        # compute endcap position error
        for u in range(args.num_endcaps):
            motion_capture_fail = np.allclose(Q[:, u], 0)
            if not motion_capture_fail:
                endcap_error_data['frame_id'].append(i)
                endcap_error_data['error (cm)'].append(distance.euclidean(Q[:, u], P_hat[:, u])*100)
                endcap_error_data['endcap_id'].append(u)


if __name__ == '__main__':
    parser = ArgumentParser("pose evalutation")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=1000, type=int)
    parser.add_argument("--num_endcaps", default=6, type=int)
    parser.add_argument("--mocap_scale", default=1000, type=int, help="scale of measured position (mm by default)")
    args = parser.parse_args()

    rod_error_data = defaultdict(list)
    endcap_error_data = defaultdict(list)

    videos = [f"{i:04d}" for i in range(1, 17)]
    for video in videos:
        eval_single_video(args, video, rod_error_data, endcap_error_data)

    rod_error_df = pd.DataFrame(data=rod_error_data)
    endcap_error_df = pd.DataFrame(data=endcap_error_data)

    print(rod_error_df.head())
    print(endcap_error_df.head())

    # rod pose error visualization
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(f"Rod Pose Error Plot")
    fig.set_size_inches(16, 12)
    axes[0, 0].grid(True)
    axes[0, 1].grid(True)
    sns.scatterplot(ax=axes[0, 0], x="frame_id", y="trans_err (cm)", hue="color", data=rod_error_df)
    sns.scatterplot(ax=axes[0, 1], x="frame_id", y="rot_err (deg)", hue="color", data=rod_error_df)
    sns.barplot(ax=axes[1, 0], x="color", y="trans_err (cm)", data=rod_error_df)
    sns.barplot(ax=axes[1, 1], x="color", y="rot_err (deg)", data=rod_error_df)
    axes[0, 0].set_ylim(0)
    axes[0, 1].set_ylim(0)
    axes[0, 0].set_title("mean translation error: %.3f(cm), std: %.3f"%(rod_error_df['trans_err (cm)'].mean(),
                                                                        rod_error_df['trans_err (cm)'].std()))
    axes[0, 1].set_title("mean rotation error: %.3f(deg), std: %.3f"%(rod_error_df['rot_err (deg)'].mean(),
                                                                      rod_error_df['rot_err (deg)'].std()))
    # plt.savefig(os.path.join(args.dataset, args.video, "error_analysis", "rod_pose_error.png"), dpi=150)

    # rod pose error visualization
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(f"Endcap Position Error Plot")
    fig.set_size_inches(10, 5)
    axes[0].grid(True)
    sns.scatterplot(ax=axes[0], x="frame_id", y="error (cm)", hue="endcap_id", data=endcap_error_df, palette="deep")
    sns.barplot(ax=axes[1], x="endcap_id", y="error (cm)", data=endcap_error_df, palette="deep")
    axes[0].set_ylim(0)
    axes[0].set_title("mean position error: %.3f, std: %.3f"%(endcap_error_df['error (cm)'].mean(),
                                                              endcap_error_df['error (cm)'].std()))
    # plt.savefig(os.path.join(args.dataset, args.video, "error_analysis", "endcap_position_error.png"), dpi=150)
    plt.show()
