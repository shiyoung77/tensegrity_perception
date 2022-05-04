import os
import importlib
import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser("pose evalutation")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--video", default="monday_roll15")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=1000, type=int)
    parser.add_argument("--num_endcaps", default=6, type=int)
    parser.add_argument("--pose_folder", default="poses", type=str)
    parser.add_argument("--mocap_scale", default=1000, type=int, help="scale of measured position (mm by default)")
    args = parser.parse_args()

    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    pos_dict = dict()
    for i in range(len(data_cfg['node_to_color'])):
        pos_dict[i] = np.load(os.path.join(args.dataset, args.video, args.pose_folder, f'{i}_pos.npy'))

    N = min(pos_dict[0].shape[0], args.end_frame)

    predicted_dist = defaultdict(list)
    measured_dist = defaultdict(list)
    mocap_dist = defaultdict(list)

    measured_errors = defaultdict(list)
    predicted_errors = defaultdict(list)

    for i in range(N):
        info_path = os.path.join(args.dataset, args.video, 'data', f'{i + args.start_frame:04d}.json')
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
                mocap_dist[sensor_id].append(la.norm(u_pos - v_pos))

            measured_errors[sensor_id].append(np.abs(mocap_dist[sensor_id][-1] - measured_dist[sensor_id][-1])*100)
            predicted_errors[sensor_id].append(np.abs(mocap_dist[sensor_id][-1] - predicted_dist[sensor_id][-1])*100)

    os.makedirs(os.path.join(args.dataset, args.video, "error_analysis"), exist_ok=True)

    fig, axes = plt.subplots(3, 3)
    fig.set_size_inches(16, 12)
    fig.suptitle(f"Cable Length Plot")
    for sensor_id in data_cfg['sensor_to_tendon']:
        row = int(sensor_id) // 3
        col = int(sensor_id) % 3
        axes[row, col].plot(range(N), np.array(predicted_dist[sensor_id])*100, label=f'{sensor_id}-predicted')
        axes[row, col].plot(range(N), np.array(measured_dist[sensor_id])*100, label=f'{sensor_id}-measured')
        axes[row, col].plot(range(N), np.array(mocap_dist[sensor_id])*100, label=f'{sensor_id}-mocap')
        axes[row, col].legend(loc='best')
        axes[row, col].set_xlabel("frame id")
        axes[row, col].set_ylabel("distance (cm)")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(args.dataset, args.video, "error_analysis", "cable_length.png"), dpi=150)

    predicted_errors_stacked = np.hstack(list(predicted_errors.values()))
    measured_errors_stacked = np.hstack(list(measured_errors.values()))
    pred_mean_err = np.nanmean(predicted_errors_stacked)
    measured_mean_err = np.nanmean(measured_errors_stacked)

    fig, axes = plt.subplots(3, 3)
    fig.set_size_inches(16, 12)
    fig.suptitle(f"Cable Length Error Plot\n"
                 f"mean predicted error of all cables (cm): {pred_mean_err:.1f}\n"
                 f"mean measured error of all cables (cm) {measured_mean_err:.1f}")
    for sensor_id in data_cfg['sensor_to_tendon']:
        row = int(sensor_id) // 3
        col = int(sensor_id) % 3
        axes[row, col].set_xlabel("frame id")
        axes[row, col].set_ylabel("error (cm)")
        axes[row, col].plot(range(N), predicted_errors[sensor_id], label=f'{sensor_id}-pred')
        axes[row, col].plot(range(N), measured_errors[sensor_id], label=f'{sensor_id}-measured')
        axes[row, col].legend(loc='best')
        axes[row, col].set_title(f"mean predicted error (cm): {np.nanmean(predicted_errors[sensor_id]):.1f}\n"
                                 f"mean measured error (cm): {np.nanmean(measured_errors[sensor_id]):.1f}",
                                 fontsize=10)
        axes[row, col].grid(True)
        axes[row, col].set_ylim(0)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(args.dataset, args.video, "error_analysis", "cable_length_error.png"), dpi=150)
    plt.show()
