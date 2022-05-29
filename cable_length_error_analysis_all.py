import os
import importlib
import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt


def eval_single_video(args, video, method, measured_errors, predicted_errors):
    print(f'{args.dataset}.{video}.config')
    data_cfg_module = importlib.import_module(f'{args.dataset}.{video}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    pose_folder = os.path.join(args.dataset, video, f'poses-{method}')
    pos_dict = dict()
    for i in range(len(data_cfg['node_to_color'])):
        pos_dict[i] = np.load(os.path.join(pose_folder, f'{i}_pos.npy'))

    N = min(pos_dict[0].shape[0], args.end_frame)

    predicted_dist = defaultdict(list)
    measured_dist = defaultdict(list)
    mocap_dist = defaultdict(list)

    for i in range(N):
        info_path = os.path.join(args.dataset, video, 'data', f'{i + args.start_frame:04d}.json')
        with open(info_path, 'r') as f:
            info = json.load(f)

        for sensor_id, (u, v) in data_cfg['sensor_to_tendon'].items():
            # measured cable length
            measured_dist[sensor_id].append(info['sensors'][str(sensor_id)]['length'] / 1000)

            # predicted cable length
            u_pos = pos_dict[u][i]
            v_pos = pos_dict[v][i]
            predicted_dist[sensor_id].append(la.norm(u_pos - v_pos))

            # ground truth cable length by motion capture
            u_pos = info['mocap'][str(u)]
            v_pos = info['mocap'][str(v)]
            if u_pos is None or v_pos is None:
                mocap_dist[sensor_id].append(np.nan)
            else:
                u_pos = np.array([u_pos['x'], u_pos['y'], u_pos['z']]) / args.mocap_scale
                v_pos = np.array([v_pos['x'], v_pos['y'], v_pos['z']]) / args.mocap_scale
                gt_length = la.norm(u_pos - v_pos)
                # mocap_dist[sensor_id].append(gt_length)
                if gt_length < 0.5:  # mocap sometimes also generates large errors
                    mocap_dist[sensor_id].append(gt_length)
                else:
                    mocap_dist[sensor_id].append(np.nan)

            measured_errors[sensor_id].append(np.abs(mocap_dist[sensor_id][-1] - measured_dist[sensor_id][-1])*100)
            predicted_errors[sensor_id].append(np.abs(mocap_dist[sensor_id][-1] - predicted_dist[sensor_id][-1])*100)


if __name__ == '__main__':
    parser = ArgumentParser("pose evalutation")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--video", default="monday_roll15")
    parser.add_argument("--method", default="proposed")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=1000, type=int)
    parser.add_argument("--num_endcaps", default=6, type=int)
    parser.add_argument("--pose_folder", default="poses", type=str)
    parser.add_argument("--mocap_scale", default=1000, type=int, help="scale of measured position (mm by default)")
    args = parser.parse_args()


    measured_errors = defaultdict(list)
    predicted_errors = defaultdict(list)

    videos = [f"{i:04d}" for i in range(1, 17)]
    for video in videos:
        eval_single_video(args, video, args.method, measured_errors, predicted_errors)

    predicted_errors_stacked = np.hstack(list(predicted_errors.values()))
    measured_errors_stacked = np.hstack(list(measured_errors.values()))
    pred_mean_err = np.nanmean(predicted_errors_stacked)
    measured_mean_err = np.nanmean(measured_errors_stacked)
    pred_std_err = np.nanstd(predicted_errors_stacked)
    measured_std_err = np.nanstd(measured_errors_stacked)

    print(f"mean predicted error of all cables (cm): {pred_mean_err:.2f}")
    print(f"std predicted error of all cables (cm): {pred_std_err:.2f}")
    print(f"mean measured error of all cables (cm) {measured_mean_err:.2f}")
    print(f"std measured error of all cables (cm) {measured_std_err:.2f}")
