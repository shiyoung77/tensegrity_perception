import os
import copy
import importlib
import json
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import numpy.linalg as la
import cv2
import open3d as o3d

import perception_utils


def visualize(data_cfg, robot_cloud, visualizer):
    visualizer.clear_geometries()
    visualizer.add_geometry(robot_cloud)

    # Due to a potential bug in Open3D, cx, cy can only be w / 2 - 0.5, h / 2 - 0.5
    # https://github.com/intel-isl/Open3D/issues/1164
    cam_intr_vis = o3d.camera.PinholeCameraIntrinsic()
    cam_intr = np.array(data_cfg['cam_intr'])
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = data_cfg['im_w'] / 2 - 0.5, data_cfg['im_h'] / 2 - 0.5
    cam_intr_vis.set_intrinsics(data_cfg['im_w'], data_cfg['im_h'], fx, fy, cx, cy)
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = cam_intr_vis
    cam_params.extrinsic = data_cfg['cam_extr']

    visualizer.get_view_control().convert_from_pinhole_camera_parameters(cam_params)
    visualizer.poll_events()
    visualizer.update_renderer()


def estimate_rod_pose_from_end_cap_centers(curr_end_cap_centers, prev_rod_pose=None):
    curr_rod_pos = (curr_end_cap_centers[0] + curr_end_cap_centers[1]) / 2
    curr_z_dir = curr_end_cap_centers[0] - curr_end_cap_centers[1]
    curr_z_dir /= la.norm(curr_z_dir)

    if prev_rod_pose is None:
        prev_rod_pose = np.eye(4)

    prev_rot = prev_rod_pose[:3, :3].copy()
    prev_z_dir = prev_rot[:, 2].copy()

    # https://math.stackexchange.com/questions/180418/
    delta_rot = perception_utils.np_rotmat_of_two_v(v1=prev_z_dir, v2=curr_z_dir)
    curr_rod_pose = np.eye(4)
    curr_rod_pose[:3, :3] = delta_rot @ prev_rot
    curr_rod_pose[:3, 3] = curr_rod_pos
    return curr_rod_pose


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--video_id", default="fabric2")
    # parser.add_argument("--video_id", default="six_cameras10")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/untethered_rod_w_end_cap.ply")
    parser.add_argument("--first_frame_id", default=0, type=int)
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video_id)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'color'))])

    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }

    rod_mesh = o3d.io.read_triangle_mesh(args.rod_mesh_file)
    rod_pcd = rod_mesh.sample_points_poisson_disk(3000)
    points = np.asarray(rod_pcd.points)
    offset = points.mean(0)
    points -= offset  # move point cloud center
    points /= data_cfg['rod_scale']  # scale points from millimeter to meter
    # o3d.visualization.draw_geometries([rod_pcd])

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=data_cfg['im_w'], height=data_cfg['im_h'])

    window_name = "observation"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 50, 500)

    for idx in tqdm(range(args.first_frame_id + 1, len(prefixes))):
        prefix = prefixes[idx]
        color_path = os.path.join(video_path, 'color', f'{prefix}.png')
        depth_path = os.path.join(video_path, 'depth', f'{prefix}.png')
        info_path = os.path.join(video_path, 'data', f'{prefix}.json')

        color_im_bgr = cv2.imread(color_path)
        color_im = cv2.cvtColor(color_im_bgr, cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
        with open(info_path, 'r') as f:
            info = json.load(f)

        robot_cloud = o3d.geometry.PointCloud()

        for color, (u, v) in data_cfg['color_to_rod'].items():
            pos_u = info['mocap'][str(u)]
            pos_v = info['mocap'][str(v)]

            if pos_u is None or pos_v is None:
                continue

            pos_u = np.array([pos_u['x'], pos_u['y'], pos_u['z']]) / 1000
            pos_v = np.array([pos_v['x'], pos_v['y'], pos_v['z']]) / 1000
            end_cap_centers = [pos_u, pos_v]
            curr_rod_pose = estimate_rod_pose_from_end_cap_centers(end_cap_centers)
            rod = copy.deepcopy(rod_pcd)
            rod.transform(curr_rod_pose)
            rod = rod.paint_uniform_color(np.array(ColorDict[color]) / 255)

            robot_cloud += rod

        # o3d.visualization.draw_geometries([robot_cloud])

        visualize(data_cfg, robot_cloud, visualizer)
        cv2.imshow(window_name, color_im_bgr)
        cv2.waitKey(1)

    cv2.destroyWindow(window_name)
