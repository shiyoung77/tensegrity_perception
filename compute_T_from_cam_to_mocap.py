"""
Estimate the transformation from the camera frame to the motion capture frame
using estimated endcap positions in camera frame and measured endcap positions
in motion capture frame

WARNING:
The result depends on pose estimation. This is not the ideal way to do it.

python >= 3.6 required
"""

import os
import json
from argparse import ArgumentParser

import numpy as np

from perception_utils import kabsch

if __name__ == '__main__':
    parser = ArgumentParser("pose evalutation")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--video", default="monday_roll15")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=1000, type=int)
    parser.add_argument("--num_endcaps", default=6, type=int)
    parser.add_argument("--mocap_scale", default=1000, type=int, help="scale of measured position (mm by default)")
    args = parser.parse_args()

    pose_folder = os.path.join(args.dataset, args.video, 'poses')
    positions = dict()
    for i in range(args.num_endcaps):
        positions[i] = np.load(os.path.join(pose_folder, f'{i}_pos.npy'))

    N = min(positions[0].shape[0], args.end_frame)
    Q = []
    P = []
    for i in range(N):
        info_path = os.path.join(args.dataset, args.video, 'data', f"{i + args.start_frame:04d}.json")
        with open(info_path, 'r') as f:
            info = json.load(f)

        for u in range(args.num_endcaps):
            pos_mc = info['mocap'][str(u)]
            if pos_mc is None:
                continue
            else:
                x = pos_mc['x'] / args.mocap_scale  # (mm) -> m
                y = pos_mc['y'] / args.mocap_scale
                z = pos_mc['z'] / args.mocap_scale
                pos_mc = np.array([x, y, z])

            pos_est = positions[u][i]  # (3,)

            Q.append(pos_mc)
            P.append(pos_est)

    Q = np.array(Q).T  # (3, num_endcaps * N)
    P = np.array(P).T  # (3, num_endcaps * N)
    T = kabsch(Q, P)

    output_path = os.path.join(args.dataset, args.video, 'cam_to_mocap.npy')
    np.save(output_path, T)
    print(T)
    print(f"Transformation matrix has been saved to {output_path}")
