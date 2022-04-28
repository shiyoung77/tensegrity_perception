import os
import json
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt


def get_config():
    cfg = dict()
    cfg['cable_length_scale'] = 1000
    cfg['end_cap_colors'] = ['red', 'green', 'blue']
    cfg['node_to_color'] = ['red', 'red', 'green', 'green', 'blue', 'blue']
    cfg['color_to_rod'] = {
        'red': [0, 1],
        'green': [2, 3],
        'blue': [4, 5]
    }
    cfg['sensor_to_tendon'] = {
        0: [3, 5],
        1: [1, 3],
        2: [1, 5],
        3: [0, 2],
        4: [0, 4],
        5: [2, 4],
        6: [2, 5],
        7: [0, 3],
        8: [1, 4]
    }
    return cfg


if __name__ == '__main__':
    parser = ArgumentParser("pose evalutation")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--video", default="april28_1")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--num_endcaps", default=6, type=int)
    parser.add_argument("--pose_folder", default="poses", type=str)
    parser.add_argument("--mocap_scale", default=1000, type=int, help="scale of measured position (mm by default)")
    args = parser.parse_args()

    N = 100
    data_cfg = get_config()

    measured_dist = defaultdict(list)
    mocap_dist = defaultdict(list)
    measured_errors = defaultdict(list)

    for i in range(N):
        info_path = os.path.join(args.dataset, args.video, 'data', '%04d.json'%(i + args.start_frame))
        with open(info_path, 'r') as f:
            info = json.load(f)

        for sensor_id, (u, v) in data_cfg['sensor_to_tendon'].items():
            # measured cable length
            measured_dist[sensor_id].append(info['sensors'][str(sensor_id)]['length'] / 1000)

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

    os.makedirs(os.path.join(args.dataset, args.video, "error_analysis"), exist_ok=True)

    fig, axes = plt.subplots(3, 3)
    fig.set_size_inches(16, 12)
    fig.suptitle("Cable Length Plot")
    for sensor_id in data_cfg['sensor_to_tendon']:
        row = int(sensor_id) // 3
        col = int(sensor_id) % 3
        axes[row, col].plot(range(N), np.array(measured_dist[sensor_id])*100, label='%d-measured'%sensor_id)
        axes[row, col].plot(range(N), np.array(mocap_dist[sensor_id])*100, label='%d-mocap'%sensor_id)
        axes[row, col].legend(loc='best')
        axes[row, col].set_xlabel("frame id")
        axes[row, col].set_ylabel("distance (cm)")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(args.dataset, args.video, "error_analysis", "cable_length.png"), dpi=150)

    measured_errors_stacked = np.hstack(list(measured_errors.values()))
    measured_mean_err = np.nanmean(measured_errors_stacked)

    fig, axes = plt.subplots(3, 3)
    fig.set_size_inches(16, 12)
    fig.suptitle("Cable Length Error Plot\n"
                 "mean measured error of all cables (cm): %.1f "%measured_mean_err)
    for sensor_id in data_cfg['sensor_to_tendon']:
        row = int(sensor_id) // 3
        col = int(sensor_id) % 3
        axes[row, col].set_xlabel("frame id")
        axes[row, col].set_ylabel("error (cm)")
        axes[row, col].plot(range(N), measured_errors[sensor_id], label='%d-measured'%sensor_id)
        axes[row, col].legend(loc='best')
        axes[row, col].set_title("mean measured error (cm): %.1f"%(np.nanmean(measured_errors[sensor_id])), fontsize=10)
        axes[row, col].grid(True)
        axes[row, col].set_ylim(0)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(args.dataset, args.video, "error_analysis", "cable_length_error.png"), dpi=150)
    plt.show()
