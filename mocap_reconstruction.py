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
from pyquaternion import Quaternion

from perception_utils import create_pcd


def visualize(data_cfg, robot_cloud, visualizer, extrinsic=None):
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
    if extrinsic is None:
        cam_params.extrinsic = np.eye(4)
    else:
        cam_params.extrinsic = extrinsic

    visualizer.get_view_control().convert_from_pinhole_camera_parameters(cam_params)
    visualizer.poll_events()
    visualizer.update_renderer()


def estimate_rod_pose_from_end_cap_centers(curr_end_cap_centers, prev_rod_pose=None):
    curr_rod_pos = (curr_end_cap_centers[0] + curr_end_cap_centers[1]) / 2
    curr_z_dir = curr_end_cap_centers[0] - curr_end_cap_centers[1]
    curr_z_dir /= la.norm(curr_z_dir)

    if prev_rod_pose is None:
        prev_rod_pose = np.eye(4)

    prev_rot = prev_rod_pose[:3, :3]
    prev_z_dir = prev_rot[:, 2]

    delta_rot = np.eye(3)
    cos_dist = prev_z_dir @ curr_z_dir
    if not np.allclose(cos_dist, 1):
        axis = np.cross(prev_z_dir, curr_z_dir)
        angle = np.arccos(cos_dist)
        delta_rot = Quaternion(axis=axis, angle=angle).rotation_matrix

    curr_rod_pose = np.eye(4)
    curr_rod_pose[:3, :3] = delta_rot @ prev_rot
    curr_rod_pose[:3, 3] = curr_rod_pos
    return curr_rod_pose


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--video", default="monday_roll15")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/untethered_rod_w_end_cap.ply")
    parser.add_argument("--rod_pcd_file", default="pcd/yale/untethered_rod_w_end_cap.pcd")
    parser.add_argument("--start_frame", default=0, type=int)
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'color'))])

    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)
    cam_intr = np.array(data_cfg['cam_intr'])

    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }

    # load transformation from camera to motion capture
    cam_to_mocap_filepath = os.path.join(args.dataset, args.video, "cam_to_mocap.npy")
    if not os.path.exists(cam_to_mocap_filepath):
        print("[WARNING] Transformation not found. Start estimating... It may not be accurate.")
        os.system(f"python compute_T_from_cam_to_mocap.py -v {args.video}")
    cam_to_mocap = np.load(cam_to_mocap_filepath)
    mocap_to_cam = la.inv(cam_to_mocap)

    assert os.path.isfile(args.rod_mesh_file) or os.path.isfile(args.rod_pcd_file), "rod geometry file is not found!"
    if os.path.isfile(args.rod_pcd_file):
        print(f"read rod pcd from {args.rod_pcd_file}")
        rod_pcd = o3d.io.read_point_cloud(args.rod_pcd_file)
    else:
        print(f"read rod triangle mesh from {args.rod_mesh_file}")
        rod_mesh = o3d.io.read_triangle_mesh(args.rod_mesh_file)
        rod_pcd = rod_mesh.sample_points_poisson_disk(5000)
        points = np.asarray(rod_pcd.points)
        offset = points.mean(0)
        points -= offset  # move point cloud center
        points /= data_cfg['rod_scale']  # scale points from millimeter to meter
        pcd_path = ".".join(args.rod_mesh_file.split('.')[:-1]) + ".pcd"
        o3d.io.write_point_cloud(pcd_path, rod_pcd)
        print(f"rod pcd file is generated and saved to {pcd_path}")

    # o3d.visualization.draw_geometries([rod_pcd])

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=data_cfg['im_w'], height=data_cfg['im_h'])

    window_name = "observation"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 50, 1000)

    os.makedirs(os.path.join(video_path, "gt_cloud"), exist_ok=True)

    for idx in tqdm(range(args.start_frame + 1, len(prefixes))):
        prefix = prefixes[idx]
        color_path = os.path.join(video_path, 'color', f'{prefix}.png')
        depth_path = os.path.join(video_path, 'depth', f'{prefix}.png')
        info_path = os.path.join(video_path, 'data', f'{prefix}.json')

        color_im_bgr = cv2.imread(color_path)
        color_im = cv2.cvtColor(color_im_bgr, cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
        scene_cloud = create_pcd(depth_im, cam_intr, color_im)
        with open(info_path, 'r') as f:
            info = json.load(f)

        robot_cloud = o3d.geometry.PointCloud()
        for color, (u, v) in data_cfg['color_to_rod'].items():
            pos_u = info['mocap'][str(u)]
            pos_v = info['mocap'][str(v)]

            if pos_u is None or pos_v is None:
                continue

            pos_u = np.array([pos_u['x'], pos_u['y'], pos_u['z']]) / data_cfg['cable_length_scale']
            pos_v = np.array([pos_v['x'], pos_v['y'], pos_v['z']]) / data_cfg['cable_length_scale']
            end_cap_centers = [pos_u, pos_v]
            curr_rod_pose = estimate_rod_pose_from_end_cap_centers(end_cap_centers)
            rod = copy.deepcopy(rod_pcd)
            rod.transform(curr_rod_pose)
            rod = rod.paint_uniform_color(np.array(ColorDict[color]) / 255)
            robot_cloud += rod

        robot_cloud.transform(mocap_to_cam)
        vis_cloud = robot_cloud + scene_cloud

        visualize(data_cfg, vis_cloud, visualizer)
        cv2.imshow(window_name, color_im_bgr)
        cv2.waitKey(1)

        if len(robot_cloud.points) > 0:
            robot_pts = np.asarray(robot_cloud.points)
            bbox = o3d.geometry.AxisAlignedBoundingBox()
            bbox.min_bound = robot_pts.min(axis=0) - 0.1
            bbox.max_bound = robot_pts.max(axis=0) + 0.1
            scene_cloud = scene_cloud.crop(bbox)
            robot_cloud += scene_cloud
            o3d.io.write_point_cloud(os.path.join(video_path, "gt_cloud", f"{idx:04d}.pcd"), robot_cloud)

    cv2.destroyWindow(window_name)
