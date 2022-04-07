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
    args = parser.parse_args()

    num_frames = len(os.listdir(os.path.join(args.dataset, args.video, 'data')))

    timestamps = []
    for i in range(num_frames):
        info_path = os.path.join(args.dataset, args.video, 'data', f'{i:04d}.json')
        with open(info_path, 'r') as f:
            info = json.load(f)

        timestamp = info['header']['secs']
        timestamps.append(timestamp)

    timestamps = np.array(timestamps)

    time_intervals = timestamps[1:] - timestamps[:-1]
    mean = time_intervals.mean()
    std = np.sqrt(np.sum((time_intervals - mean)**2) / (num_frames - 1))
    plt.plot(range(num_frames - 1), time_intervals)
    plt.xlabel('frame id')
    plt.ylabel('time interval (s)')
    plt.title(f'{mean = :.3f}s, {std = :.3f}s')
    plt.grid(True)
    plt.show()
