import os
import copy
import importlib
import argparse

import numpy as np
import open3d as o3d


def visualize(visualizer, data_cfg, vis_cloud):
    visualizer.clear_geometries()
    visualizer.add_geometry(vis_cloud)

    # Due to a potential bug in Open3D, cx, cy can only be w / 2 - 0.5, h / 2 - 0.5
    # https://github.com/intel-isl/Open3D/issues/1164
    cam_intr_vis = o3d.camera.PinholeCameraIntrinsic()
    cam_intr = np.array(data_cfg['cam_intr'])
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = data_cfg['im_w'] / 2 - 0.5, data_cfg['im_h'] / 2 - 0.5
    cam_intr_vis.set_intrinsics(data_cfg['im_w'], data_cfg['im_h'], fx, fy, cx, cy)
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = cam_intr_vis
    cam_params.extrinsic = np.eye(4)
    # cam_params.extrinsic = data_cfg['cam_extr']

    visualizer.get_view_control().convert_from_pinhole_camera_parameters(cam_params)
    visualizer.poll_events()
    visualizer.update_renderer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("-v", "--video_id", default="kun21step_7")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/untethered_rod_w_end_cap.ply")
    parser.add_argument("--first_frame_id", default=0, type=int)
    # parser.add_argument("--first_frame_id", default=20, type=int)
    parser.add_argument("--window_size", default=5, type=int)
    args = parser.parse_args()

    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    rod_mesh = o3d.io.read_triangle_mesh(args.rod_mesh_file)
    rod_pcd = rod_mesh.sample_points_poisson_disk(3000)
    points = np.asarray(rod_pcd.points)
    offset = points.mean(0)
    points -= offset  # move point cloud center
    points /= 1000  # scale points from millimeter to meter
    # o3d.visualization.draw_geometries([rod_pcd])

    color_poses = dict()
    smoothed_color_poses = dict()
    pose_output_folder = os.path.join(args.dataset, args.video_id, 'smoothed_poses')
    if not os.path.isdir(pose_output_folder):
        os.makedirs(pose_output_folder)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=data_cfg['im_w'], height=data_cfg['im_h'])

    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }

    for color in data_cfg['end_cap_colors']:
        color_poses[color] = np.load(os.path.join(args.dataset, args.video_id, 'poses', f'{color}.npy'))
        smoothed_color_poses[color] = np.copy(color_poses[color])

    smoothed_node_poses = dict()
    for node_id in range(data_cfg['num_rods'] * 2):
        smoothed_node_poses[node_id] = np.load(os.path.join(args.dataset, args.video_id, 'poses', f'{node_id}_pos.npy'))

    os.makedirs(os.path.join(args.dataset, args.video_id, "smoothed_estimation"), exist_ok=True)
    # os.makedirs(os.path.join(args.dataset, args.video_id, "smoothed_estimation_cloud"), exist_ok=True)

    N = len(color_poses['red'])
    t = (args.window_size - 1) // 2
    for i in range(t, N - t):
        # scene_pcd_path = os.path.join(args.dataset, args.video_id, 'scene_cloud', f"{(args.first_frame_id + i):04d}.ply")
        # scene_cloud = o3d.io.read_point_cloud(scene_pcd_path)
        # vis_cloud = scene_cloud
        vis_cloud = o3d.geometry.PointCloud()

        for color in data_cfg['end_cap_colors']:
            mean_rot_hat = np.zeros((3, 3))
            mean_trans_hat = np.zeros(3)
            for j in range(-t, t + 1):
                pose = color_poses[color][i + j]
                rot = pose[:3, :3]
                pos = pose[:3, 3]
                mean_rot_hat += rot
                mean_trans_hat += pos
            # mean_rot_hat /= window_size
            U, D, V_h = np.linalg.svd(mean_rot_hat, full_matrices=True)
            assert np.allclose((U * D) @ V_h, mean_rot_hat)
            mean_rot = U @ V_h
            mean_trans = mean_trans_hat / args.window_size

            mean_pose = np.eye(4)
            mean_pose[:3, :3] = mean_rot
            mean_pose[:3, 3] = mean_trans
            smoothed_color_poses[color][i] = mean_pose

            rod_pcd_copy = copy.deepcopy(rod_pcd)
            rod_pcd_copy.transform(mean_pose)
            rod_pcd_copy = rod_pcd_copy.paint_uniform_color(np.array(ColorDict[color]) / 255)
            vis_cloud += rod_pcd_copy

        for color, (u, v) in data_cfg['color_to_rod'].items():
            smoothed_pose = smoothed_color_poses[color][i]
            smoothed_node_poses[u][i] = smoothed_pose[:3, 3] + smoothed_pose[:3, 2] * data_cfg['rod_length'] / 2
            smoothed_node_poses[v][i] = smoothed_pose[:3, 3] - smoothed_pose[:3, 2] * data_cfg['rod_length'] / 2

        visualize(visualizer, data_cfg, vis_cloud)
        # o3d.io.write_point_cloud(os.path.join(args.dataset, args.video_id, "smoothed_estimation_cloud", f"{i:04d}.ply"), vis_cloud)
        visualizer.capture_screen_image(os.path.join(args.dataset, args.video_id, "smoothed_estimation", f"{i:04d}.png"))
        # visualizer.capture_screen_image(os.path.join(args.dataset, args.video_id, "without_physical", f"{i:04d}.png"))

    for color in data_cfg['end_cap_colors']:
        np.save(os.path.join(pose_output_folder, f'{color}.npy'), smoothed_color_poses[color])

    for node_id in range(data_cfg['num_rods']*2):
        np.save(os.path.join(pose_output_folder, f'{node_id}_pos.npy'), smoothed_node_poses[node_id])
