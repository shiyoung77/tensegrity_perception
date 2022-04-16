"""
compute statistics of data captured from the motion capture system
"""
from collections import defaultdict
import os
import importlib
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--video_id", default="monday_roll15")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/untethered_rod_w_end_cap.ply")
    parser.add_argument("--rod_pcd_file", default="pcd/yale/untethered_rod_w_end_cap.pcd")
    parser.add_argument("--start_frame", default=0, type=int)
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video_id)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'data'))])
    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    counter = defaultdict(int)
    counter["num_frames"] = len(prefixes) - args.start_frame

    for idx in range(args.start_frame, len(prefixes)):
        prefix = prefixes[idx]
        info_path = os.path.join(video_path, 'data', f'{prefix}.json')
        with open(info_path, 'r') as f:
            info = json.load(f)

        for color, (u, v) in data_cfg['color_to_rod'].items():
            if info['mocap'][str(u)] is not None:
                counter[u] += 1
            if info['mocap'][str(v)] is not None:
                counter[v] += 1

        for sensor_id, (u, v) in data_cfg['sensor_to_tendon'].items():
            if info['mocap'][str(u)] is not None and info['mocap'][str(v)] is not None:
                counter["%d,%d"%(u, v)] += 1

    node_valid_ratios = dict()
    for color, (u, v) in data_cfg['color_to_rod'].items():
        node_valid_ratios[u] = counter[u] / counter['num_frames']
        node_valid_ratios[v] = counter[v] / counter['num_frames']
        print("endcap %d, %d / %d = %.3f"%(u, counter[u], counter['num_frames'], node_valid_ratios[u]))
        print("endcap %d, %d / %d = %.3f"%(v, counter[v], counter['num_frames'], node_valid_ratios[v]))

    print('--------------------------------------------------')
    bar_valid_ratios = dict()
    for sensor_id, (u, v) in data_cfg['sensor_to_tendon'].items():
        valid_ratio = counter["%d,%d"%(u, v)] / counter['num_frames']
        bar_valid_ratios["(%d,%d)"%(u, v)] = valid_ratio
        print("endcap distances (%d, %d), %d / %d = %.3f"%(u, v, counter["%d,%d"%(u, v)], counter['num_frames'], valid_ratio))

    # save stats
    output_path = os.path.join(video_path, 'error_analysis', 'mocap_stats.json')
    with open(output_path, 'w') as f:
        json.dump(counter, f, indent=4)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    axes[0].bar([str(i) for i in node_valid_ratios.keys()], node_valid_ratios.values())
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)
    axes[0].set_title("endcap position valid ratio")
    axes[1].bar(bar_valid_ratios.keys(), bar_valid_ratios.values())
    axes[1].grid(True)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("endcap distances valid ratio")
    plt.show()
