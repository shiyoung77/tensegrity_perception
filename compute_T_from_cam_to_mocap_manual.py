"""
Estimate the transformation from the camera frame to the motion capture frame
using manually labeled endcap positions in camera frame and measured endcap
positions in motion capture frame

python >= 3.6 required
"""

import os
import importlib
import json
from argparse import ArgumentParser

import numpy as np
import cv2

from perception_utils import kabsch


clicked_pts = []

def click_callback(event, x, y, flags, param):
    global clicked_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pts.append([x, y])
        print("clicked point (%d, %d)"%(x, y))
        print(clicked_pts)


if __name__ == '__main__':
    parser = ArgumentParser("pose evalutation")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--video", default="shiyang3")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--mocap_scale", default=1000, type=int, help="scale of measured position (mm by default)")
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'data'))])
    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)
    cam_intr = np.array(data_cfg['cam_intr'])

    # manually label the first frame
    color_im_path = os.path.join(video_path, 'labels.png')
    color_im = cv2.imread(color_im_path, cv2.IMREAD_COLOR)
    cv2.imshow("image", color_im)
    cv2.setMouseCallback("image", click_callback)
    while True:
        cv2.imshow("image", color_im)
        key = cv2.waitKey(1)
        if key == ord('r'):
            x, y = clicked_pts.pop()
            print("removed point (%d, %d)"%(x, y))
            print("remaining clicked points:")
            print(clicked_pts)
        elif key == ord('q'):
            print("====================")
            print("clicked points:")
            print(clicked_pts)
            break

    labels = [6, 7, 8, 9]

    P = []  # cam points
    Q = []  # mocap points

    for idx in range(args.start_frame, len(prefixes)):
        prefix = prefixes[idx]
        depth_im_path = os.path.join(video_path, 'depth', f'{prefix}.png')
        depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED) / data_cfg['depth_scale']
        info_path = os.path.join(video_path, 'data', f'{prefix}.json')
        with open(info_path, 'r') as f:
            info = json.load(f)

        for label, (u, v) in zip(labels, clicked_pts):
            # get point position in the motion capture coordinate
            depth = depth_im[v, u]
            pos_mc = info['mocap'][str(label)]
            if pos_mc is None or depth == 0:
                continue

            x = pos_mc['x'] / args.mocap_scale  # (mm) -> m
            y = pos_mc['y'] / args.mocap_scale
            z = pos_mc['z'] / args.mocap_scale
            pos_mc = np.array([x, y, z])
            Q.append(pos_mc)

            # get point position in the camera coordinate
            x = depth * (u - cam_intr[0, 2]) / cam_intr[0, 0]
            y = depth * (v - cam_intr[1, 2]) / cam_intr[1, 1]
            z = depth
            pos_est = np.array([x, y, z])
            P.append(pos_est)

    P = np.vstack(P).T
    Q = np.vstack(Q).T
    T = kabsch(Q, P)
    print(T)

    output_path = os.path.join(args.dataset, args.video, 'cam_to_mocap.npy')
    np.save(output_path, T)
    print(f"Transformation matrix has been saved to {output_path}")
