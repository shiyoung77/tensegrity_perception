import os
import json

import numpy as np
import cv2

from perception_utils import vis_pcd, create_pcd


def main():
    dataset = "/mnt/evo/dataset/tensegrity/2023-04-18_17-31-48"
    with open(os.path.join(dataset, 'config.json'), 'r') as f:
        cam_info = json.load(f)
    cam_intr = cam_info['cam_intr']
    cam_extr = cam_info['cam_extr']
    depth_scale = cam_info['depth_scale']

    prefix = 0
    color_path = os.path.join(dataset, 'color', f'{prefix:04d}.png')
    depth_path = os.path.join(dataset, 'depth', f'{prefix:04d}.png')

    color_im = cv2.cvtColor(cv2.imread(color_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
    pcd = create_pcd(depth_im, cam_intr, color_im, depth_trunc=3)
    vis_pcd(pcd)

    pcd.transform(cam_extr)
    vis_pcd(pcd)


if __name__ == "__main__":
    main()
