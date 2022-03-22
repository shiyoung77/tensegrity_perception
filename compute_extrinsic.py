"""
estimate the transoformation between the camera coordinates and motion capture coordinates
"""

import os
import importlib
import json
import torch
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


dataset = 'dataset'
# video_id = 'socks6'
video_id = 'monday_roll15'
pose_folder = os.path.join(dataset, video_id, 'poses')

# read data config
data_cfg_module = importlib.import_module(f'{dataset}.{video_id}.config')
data_cfg = data_cfg_module.get_config(read_cfg=True)

positions = dict()
for i in range(6):
    positions[i] = np.load(os.path.join(pose_folder, f'{i}_pos.npy'))

# pts_est = np.zeros((6, 3))
# for i in range(6):
#     pts_est[i] = positions[i][0]

start_frame_id = 0
# pts_mc = np.zeros((6, 3))
# with open(os.path.join(dataset, video_id, 'data', f"{start_frame_id:04d}.json"), 'r') as f:
#     info = json.load(f)
# for i in range(6):
#     x = info['mocap'][str(i)]['x'] / 1000
#     y = info['mocap'][str(i)]['y'] / 1000
#     z = info['mocap'][str(i)]['z'] / 1000
#     pts_mc[i] = [x, y, z]

# Q = torch.from_numpy(pts_mc).T
# P = torch.from_numpy(pts_est).T
# P_mean = P.mean(1, keepdims=True)  # (3, 1)
# Q_mean = Q.mean(1, keepdims=True)  # (3, 1)

# H = (Q - Q_mean) @ (P - P_mean).T
# U, D, V = torch.svd(H)
# R = U @ V.T
# t = Q_mean - R @ P_mean

# T = np.eye(4)
# T[:3, :3] = R
# T[:3, 3:4] = t

# P_hat = R @ P + t

# error = 0
# for color, (u, v) in data_cfg['color_to_rod'].items():
#     rod_error = torch.norm((P_hat[:, u] + P_hat[:, v]) / 2 - (Q[:, u] + Q[:, v]) / 2)
#     print(rod_error)
#     error += rod_error
# error /= 3
# print(error)


Q = []
P = []

for i in range(10):
    with open(os.path.join(dataset, video_id, 'data', f"{i + start_frame_id:04d}.json"), 'r') as f:
        info = json.load(f)

    for j in range(6):
        pos_mc = info['mocap'][str(j)]
        if pos_mc is None:
            continue
        else:
            x = pos_mc['x'] / 1000
            y = pos_mc['y'] / 1000
            z = pos_mc['z'] / 1000
            pos_mc = np.array([x, y, z])

        pos_est = positions[j][i]  # (3,)

        Q.append(pos_mc)
        P.append(pos_est)

Q = np.array(Q).T
P = np.array(P).T
Q = torch.from_numpy(Q)
P = torch.from_numpy(P)
P_mean = P.mean(1, keepdims=True)  # (3, 1)
Q_mean = Q.mean(1, keepdims=True)  # (3, 1)

H = (Q - Q_mean) @ (P - P_mean).T
U, D, V = torch.svd(H)
R = U @ V.T
t = Q_mean - R @ P_mean

P_hat = R @ P + t
error = 0
for color, (u, v) in data_cfg['color_to_rod'].items():
    rod_error = torch.norm((P_hat[:, u] + P_hat[:, v]) / 2 - (Q[:, u] + Q[:, v]) / 2)
    error += rod_error
error /= 3
print(error)

R = torch.tensor([[-0.7530,  0.6580,  0.0088],
            [-0.0320, -0.0500,  0.9982],
            [ 0.6572,  0.7514,  0.0587]], dtype=torch.float64)
t = torch.tensor([[-0.5749],
            [ 1.2045],
            [ 0.1435]], dtype=torch.float64)

print("rotation matrix")
print(R)
print("translation")
print(t)

########################################################################################
errors = {color: [] for color in data_cfg['color_to_rod']}
for i in range(111):
    pts_est = np.zeros((6, 3))
    for j in range(6):
        pts_est[j] = positions[j][i]

    pts_mc = np.zeros((6, 3))
    with open(os.path.join(dataset, video_id, 'data', f"{i + start_frame_id:04d}.json"), 'r') as f:
        info = json.load(f)
    for j in range(6):
        pos = info['mocap'][str(j)]
        if pos is None:
            pts_mc[j] = [0, 0, 0]
        else:
            x = pos['x'] / 1000
            y = pos['y'] / 1000
            z = pos['z'] / 1000
            pts_mc[j] = [x, y, z]

    Q = pts_mc.T
    P = pts_est.T
    P_hat = R @ P + t

    for j, [color, (u, v)] in enumerate(data_cfg['color_to_rod'].items()):
        if np.all(Q[:, u] == 0) or np.all(Q[:, v] == 0):
            errors[color].append(0)
        else:
            rod_error = la.norm((P_hat[:, u] + P_hat[:, v]) / 2 - (Q[:, u] + Q[:, v]) / 2)
            # a = P_hat[:, u] - P_hat[:, v]
            # a /= la.norm(a)
            # b = Q[:, u] - Q[:, v]
            # b /= la.norm(b)
            # rod_error = 180 * np.arccos(a @ b) / np.pi
            errors[color].append(rod_error)

avg_error = 0
count = 0
valid_errors = []
for color, error in errors.items():
    error = np.array(error)
    count += np.sum(error != 0)
    avg_error += np.sum(error[error != 0])
    valid_errors.append(error[error != 0])

print(f"{count = }")
# print(avg_error / count)
valid_errors = np.hstack(valid_errors)
print(valid_errors.mean())
print(np.sqrt(valid_errors.var()))

N = 111

# plt.scatter(np.arange(210) + start_frame_id, errors['red'], label='red', c=np.array([[1, 0, 0]]))
# plt.scatter(np.arange(210) + start_frame_id, errors['blue'], label='blue', c=np.array([[0, 0, 1]]))
# plt.scatter(np.arange(210) + start_frame_id, errors['green'], label='green', c=np.array([[0, 1, 0]]))
plt.scatter(np.arange(N), errors['red'], label='red', c=np.array([[1, 0, 0]]))
plt.scatter(np.arange(N), errors['blue'], label='blue', c=np.array([[0, 0, 1]]))
plt.scatter(np.arange(N), errors['green'], label='green', c=np.array([[0, 1, 0]]))
plt.xlabel("frame id")
plt.ylabel("rotation error (deg)")
plt.grid(True)
plt.title("Rotation Error")
plt.legend(title='rod end cap color')
plt.show()
