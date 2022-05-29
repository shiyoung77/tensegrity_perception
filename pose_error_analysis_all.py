# Anaconda recommended
# python >= 3.6 required

import os
import importlib
import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as distance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from compute_T_from_cam_to_mocap import compute_extrinsic


def mocap_is_valid(data_cfg, info):
    for u, v in data_cfg['sensor_to_tendon'].values():
        u_pos = info['mocap'][str(u)]
        v_pos = info['mocap'][str(v)]
        if u_pos is not None and v_pos is not None:
            u_pos = np.array([u_pos['x'], u_pos['y'], u_pos['z']]) / args.mocap_scale
            v_pos = np.array([v_pos['x'], v_pos['y'], v_pos['z']]) / args.mocap_scale
            gt_length = la.norm(u_pos - v_pos)
            if gt_length > 0.5:
                return False
    return True


def eval_single_video(args, video, method, rod_error_data, endcap_error_data):
    print(f'{args.dataset}.{video}.config')
    data_cfg_module = importlib.import_module(f'{args.dataset}.{video}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    pose_folder = os.path.join(args.dataset, video, f'poses-{method}')
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
    # T = compute_extrinsic(args, video, method)
    R = T[:3, :3]
    t = T[:3, 3:4]

    two_five_total = 0
    two_five_inlier = 0

    for i in range(N):  # for each frame
        pts_est = np.zeros((args.num_endcaps, 3))
        for u in range(args.num_endcaps):
            pts_est[u] = positions[u][i]

        pts_mc = np.zeros((args.num_endcaps, 3))
        with open(os.path.join(args.dataset, video, 'data', f"{i + args.start_frame:04d}.json"), 'r') as f:
            info = json.load(f)

        if not mocap_is_valid(data_cfg, info):
            continue

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
        all_visible = True
        predicted_rod_pos = []
        measured_rod_pos = []

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

                predicted_rod_pos.append(a)
                measured_rod_pos.append(b)

                # rotation error
                a = P_hat[:, u] - P_hat[:, v]
                b = Q[:, u] - Q[:, v]
                rot_err = np.rad2deg(np.arccos(1 - distance.cosine(a, b)))
                rod_error_data['rot_err (deg)'].append(rot_err)

                two_five_total += 1
                if trans_err < 0.02 and rot_err < 5:
                    two_five_inlier += 1
            else:
                all_visible = False

        # compute center of mass error
        if all_visible:
            assert len(predicted_rod_pos) == 3 and len(measured_rod_pos) == 3, "!!!"
            predicted_CoM = np.stack(predicted_rod_pos).mean(0)
            measured_CoM = np.stack(measured_rod_pos).mean(0)
            error_CoM = distance.euclidean(predicted_CoM, measured_CoM)
            rod_error_data['error_CoM'].append(error_CoM)

        # compute endcap position error
        for u in range(args.num_endcaps):
            motion_capture_fail = np.allclose(Q[:, u], 0)
            if not motion_capture_fail:
                endcap_error_data['frame_id'].append(i)
                endcap_error_data['error (cm)'].append(distance.euclidean(Q[:, u], P_hat[:, u])*100)
                endcap_error_data['endcap_id'].append(u)

    rod_error_data['two_five_total'].append(two_five_total)
    rod_error_data['two_five_inlier'].append(two_five_inlier)

if __name__ == '__main__':
    parser = ArgumentParser("pose evalutation")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--method", default="proposed")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=1000, type=int)
    parser.add_argument("--num_endcaps", default=6, type=int)
    parser.add_argument("--mocap_scale", default=1000, type=int, help="scale of measured position (mm by default)")
    args = parser.parse_args()

    rod_error_data = defaultdict(list)
    endcap_error_data = defaultdict(list)

    videos = [f"{i:04d}" for i in range(1, 17)]
    for video in videos:
        eval_single_video(args, video, args.method, rod_error_data, endcap_error_data)

    print(f"{rod_error_data['two_five_total'] = }")
    print(f"{rod_error_data['two_five_inlier'] = }")
    print(f"ratio = {sum(rod_error_data['two_five_inlier']) / sum(rod_error_data['two_five_total'])}")
    print(f"{np.array(rod_error_data['error_CoM']).mean() * 100 = }")
    print(f"{np.array(rod_error_data['error_CoM']).std() * 100 = }")
    print(f"{np.array(rod_error_data['trans_err (cm)']).mean() = }")
    print(f"{np.array(rod_error_data['trans_err (cm)']).std() = }")
    print(f"{np.array(rod_error_data['rot_err (deg)']).mean() = }")
    print(f"{np.array(rod_error_data['rot_err (deg)']).std() = }")
