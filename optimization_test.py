import os
import time
import copy
import importlib
import pprint
import json
import time

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm.contrib import tenumerate

'''
# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},  # >=
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},  # >=
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})  # >=
bnds = ((0, None), (0, None))

tic = time.time()
res = minimize(fun, (1321, -13123), method='SLSQP', bounds=bnds, constraints=cons)
print(time.time() - tic)
print(res)
'''

if __name__ == '__main__':
    dataset = 'dataset'
    video_id = "crawling_sim"
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(dataset, video_id, 'color'))])

    data_cfg_module = importlib.import_module(f'{dataset}.{video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    pose_output_folder = os.path.join(dataset, video_id, "poses")
    pos_dict = dict()
    for i in range(len(data_cfg['node_to_color'])):
        pos_dict[i] = np.load(os.path.join(pose_output_folder, f'{i}_pos.npy'))

    predicted_dist = dict()
    measured_dist = dict()
    for sensor_id in data_cfg['sensor_to_tendon']:
        predicted_dist[sensor_id] = []
        measured_dist[sensor_id] = []

    N = len(prefixes)
    first_frame_id = 20
    for idx in range(N - first_frame_id):
        prefix = prefixes[idx + first_frame_id]
        info_path = os.path.join(dataset, video_id, 'data', f'{prefix}.json')
        with open(info_path, 'r') as f:
            info = json.load(f)

        for sensor_id, (u, v) in data_cfg['sensor_to_tendon'].items():
            measured_dist[sensor_id].append(info['sensors'][str(sensor_id)]['length'] / 100)

            u_pos = pos_dict[u][idx]
            v_pos = pos_dict[v][idx]
            predicted_dist[sensor_id].append(la.norm(u_pos - v_pos))

    fig, axes = plt.subplots(3, 3)
    for sensor_id in data_cfg['sensor_to_tendon']:
        row = int(sensor_id) // 3
        col = int(sensor_id) % 3
        axes[row, col].plot(range(N - first_frame_id), predicted_dist[sensor_id], label=f'{sensor_id}-predicted')
        axes[row, col].plot(range(N - first_frame_id), measured_dist[sensor_id], label=f'{sensor_id}-measured')
        axes[row, col].legend(loc='upper left')
    plt.show()
