import os
import json
import open3d as o3d
import numpy as np
import cv2
from pathlib import Path

from perception_utils import create_pcd, vis_pcd


def main():
    dataset = Path("~/dataset/tensegrity/limbo_test_11").expanduser()
    with open(os.path.join(dataset, 'config.json'), 'r') as f:
        cam_info = json.load(f)
    cam_intr = np.array(cam_info['cam_intr'])
    print(cam_intr)
    cam_extr = cam_info['cam_extr']
    depth_scale = cam_info['depth_scale']
    depth_trunc = cam_info['depth_trunc']

    prefix = 200
    color_path = dataset / 'color' / f'{prefix:04d}.png'
    depth_path = dataset / 'depth' / f'{prefix:04d}.png'

    color_im = cv2.cvtColor(cv2.imread(str(color_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
    pcd = create_pcd(depth_im, cam_intr, color_im, depth_trunc=depth_trunc)
    vis_pcd(pcd)

    pcd.transform(cam_extr)
    vis_pcd(pcd)
    # o3d.visualization.draw_geometries_with_editing([pcd])


if __name__ == "__main__":
    main()
