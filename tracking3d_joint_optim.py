import os
import time
import copy
import importlib
import pprint
import json
from argparse import ArgumentParser

import numpy as np
import numpy.linalg as la
import cv2
import networkx as nx
import open3d as o3d
from pyrender.constants import RenderFlags
from pyrender.renderer import Renderer
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm
import trimesh
import pyrender
import torch
from torch.optim import SGD
from lietorch import SO3, SE3, LieGroupParameter  # https://github.com/princeton-vl/lietorch

import perception_utils


class Tracker:

    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }


    def __init__(self, data_cfg, rod_pcd=None, rod_mesh=None):
        self.data_cfg = data_cfg
        pprint.pprint(self.data_cfg)
        self.initialized = False

        self.rod_mesh = rod_mesh
        self.rod_pcd = copy.deepcopy(rod_pcd)
        self.rod_length = self.data_cfg['rod_length']

        H, W = self.data_cfg['im_h'], self.data_cfg['im_w']
        self.renderer = pyrender.OffscreenRenderer(W, H)

        # initialize graph
        self.G = nx.Graph()
        self.G.graph['rods'] = [(u, v) for (u, v) in self.data_cfg['color_to_rod'].values()]
        self.G.graph['tendons'] = [(u, v) for (u, v) in self.data_cfg['sensor_to_tendon'].values()]

        # add end cap (marker) node to graph
        for node, color in enumerate(self.data_cfg['node_to_color']):
            self.G.add_node(node)
            self.G.nodes[node]['color'] = color
            self.G.nodes[node]['pos_list'] = []

        # add tendon edge to graph
        for sensor_id, (u, v) in self.data_cfg['sensor_to_tendon'].items():
            self.G.add_edge(u, v)
            self.G.edges[u, v]['sensor_id'] = sensor_id
            self.G.edges[u, v]['type'] = 'tendon'

        # add rod edge to graph
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            self.G.add_edge(u, v)
            self.G.edges[u, v]['type'] = 'rod'
            self.G.edges[u, v]['length'] = self.data_cfg['rod_length']
            self.G.edges[u, v]['color'] = color
            self.G.edges[u, v]['pose_list'] = []


    def initialize(self, color_im: np.ndarray, depth_im: np.ndarray, info: dict,
                   visualize: bool = True, compute_hsv: bool = True):

        scene_pcd = perception_utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im,
                                                depth_trunc=self.data_cfg['depth_trunc'])
        if 'cam_extr' not in self.data_cfg:
            plane_frame, _ = perception_utils.plane_detection_ransac(scene_pcd, inlier_thresh=0.005,
                                                                     visualize=visualize)
            self.data_cfg['cam_extr'] = np.round(plane_frame, decimals=3)
        else:
            plane_frame = self.data_cfg['cam_extr']
        scene_pcd.transform(la.inv(self.data_cfg['cam_extr']))

        if 'init_end_cap_rois' not in self.data_cfg:
            color_im_bgr = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
            end_cap_rois = dict()
            for u in self.G.nodes:
                color = self.G.nodes[u]['color']
                window_name = f'Select #{u} end cap ({color})'
                end_cap_roi = cv2.selectROI(window_name, color_im_bgr, fromCenter=False, showCrosshair=False)
                cv2.destroyWindow(window_name)
                end_cap_rois[str(u)] = end_cap_roi
            self.data_cfg['init_end_cap_rois'] = end_cap_rois

        # visualize init bboxes
        color_im_vis = color_im.copy()
        for u in self.G.nodes:
            color = self.G.nodes[u]['color']
            roi = self.data_cfg['init_end_cap_rois'][str(u)]
            pt1 = (int(roi[0]), int(roi[1]))
            pt2 = (pt1[0] + int(roi[2]), pt1[1] + int(roi[3]))
            cv2.rectangle(color_im_vis, pt1, pt2, Tracker.ColorDict[color], 2)
        color_im_vis = cv2.cvtColor(color_im_vis, cv2.COLOR_RGB2BGR)

        if visualize:
            cv2.imshow("init bboxes", color_im_vis)
            cv2.waitKey(0)
            cv2.destroyWindow("init bboxes")

        # calculate hsv histograms
        if compute_hsv:
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                h_hist = np.zeros((256, 1))
                s_hist = np.zeros((256, 1))
                v_hist = np.zeros((256, 1))
                for i in (u, v):
                    roi = self.data_cfg['init_end_cap_rois'][str(i)]
                    pt1 = (int(roi[0]), int(roi[1]))
                    pt2 = (pt1[0] + int(roi[2]), pt1[1] + int(roi[3]))
                    cropped_im = color_im[pt1[1]:pt2[1], pt1[0]:pt2[0]].copy()
                    cropped_im_hsv = cv2.cvtColor(cropped_im, cv2.COLOR_RGB2HSV)
                    h_hist += cv2.calcHist([cropped_im_hsv], [0], None, [256], [0, 256])
                    s_hist += cv2.calcHist([cropped_im_hsv], [1], None, [256], [0, 256])
                    v_hist += cv2.calcHist([cropped_im_hsv], [2], None, [256], [0, 256])

                if visualize:
                    plt.plot(h_hist, label='h', color='r')
                    plt.plot(s_hist, label='s', color='g')
                    plt.plot(v_hist, label='v', color='b')
                    plt.xlim([0, 256])
                    plt.title(color)
                    plt.legend()
                    plt.show()

        # estimate init rod transformation
        color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)
        H, W, _ = color_im_hsv.shape

        complete_obs_color = np.zeros_like(color_im)
        complete_obs_depth = np.zeros_like(depth_im)
        complete_obs_mask = np.zeros((H, W), dtype=np.bool8)

        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            end_cap_centers = []
            obs_depth_im = np.zeros_like(depth_im)

            for node in (u, v):
                roi = self.data_cfg['init_end_cap_rois'][str(node)]
                pt1 = (int(roi[0]), int(roi[1]))
                pt2 = (pt1[0] + int(roi[2]), pt1[1] + int(roi[3]))

                mask = np.zeros((H, W), dtype=np.bool8)
                mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = True
                hsv_mask = cv2.inRange(color_im_hsv,
                                       lowerb=tuple(self.data_cfg['hsv_ranges'][color][0]),
                                       upperb=tuple(self.data_cfg['hsv_ranges'][color][1])).astype(np.bool8)
                if color == 'red':
                    hsv_mask |= cv2.inRange(color_im_hsv, lowerb=(0, 150, 50), upperb=(20, 255, 255)).astype(np.bool8)
                mask &= hsv_mask

                obs_depth_im[mask] = depth_im[mask]
                masked_depth_im = depth_im * mask
                end_cap_pcd = perception_utils.create_pcd(masked_depth_im, self.data_cfg['cam_intr'],
                                                          cam_extr=self.data_cfg['cam_extr'])

                end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)
                end_cap_centers.append(end_cap_center)

                complete_obs_mask |= mask

            # compute rod pose given end cap centers
            init_pose = self.estimate_rod_pose_from_end_cap_centers(end_cap_centers)
            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)

            # icp refinement
            mesh_node = pyrender.Node(name='mesh', mesh=self.rod_mesh, matrix=np.eye(4))
            rod_pose = self.projective_icp_cuda(obs_depth_im, [mesh_node], [init_pose], max_iter=30, max_distance=0.02,
                                                early_stop_thresh=0.0015, verbose=False)[0]

            self.G.edges[u, v]['pose_list'].append(rod_pose)
            self.G.nodes[u]['pos_list'].append(rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[v]['pos_list'].append(rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2)

        complete_obs_color[complete_obs_mask] = color_im[complete_obs_mask]
        complete_obs_depth[complete_obs_mask] = depth_im[complete_obs_mask]

        # self.constrained_optimization(info, balance_factor=0.8)
        # self.rigid_finetune(complete_obs_depth)

        if visualize:
            _, axes = plt.subplots(1, 3)
            axes[0].imshow(color_im)
            axes[1].imshow(complete_obs_color)
            axes[2].imshow(complete_obs_depth)
            plt.show()

            robot_cloud = o3d.geometry.PointCloud()
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                rod_pcd = copy.deepcopy(self.rod_pcd)
                rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                rod_pose = self.G.edges[u, v]['pose_list'][-1]
                rod_pcd.transform(rod_pose)
                robot_cloud += rod_pcd
            o3d.visualization.draw_geometries([scene_pcd, robot_cloud])

        self.initialized = True


    def estimate_rod_pose_from_end_cap_centers(self, curr_end_cap_centers, prev_rod_pose=None):
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


    def update(self, color_im, depth_im, info):
        assert self.initialized, "[Error] You must first initialize the tracker!"

        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]
        cam_extr = np.asarray(self.data_cfg['cam_extr'])

        color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)
        depth_im_gpu = torch.from_numpy(depth_im).cuda()

        # get observed pts for each node (end cap)
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            hsv_mask = cv2.inRange(color_im_hsv,
                                   lowerb=tuple(self.data_cfg['hsv_ranges'][color][0]),
                                   upperb=tuple(self.data_cfg['hsv_ranges'][color][1])).astype(np.bool8)
            if color == 'red':
                hsv_mask |= cv2.inRange(color_im_hsv, lowerb=(0, 150, 50), upperb=(20, 255, 255)).astype(np.bool8)

            hsv_mask_gpu = torch.from_numpy(hsv_mask).cuda()
            indices = torch.nonzero(hsv_mask_gpu)
            ys = indices[:, 0]
            xs = indices[:, 1]
            Z_obs = depth_im_gpu[ys, xs]
            X_obs = (xs - cx) / fx * Z_obs
            Y_obs = (ys - cy) / fy * Z_obs
            pts_obs = torch.stack([X_obs, Y_obs, Z_obs]).T  # (N, 3)

            # crop observation
            r = 0.03
            for node in (u, v):
                prev_pos_world = self.G.nodes[node]['pos_list'][-1]
                prev_pos_cam = cam_extr[:3, :3] @ prev_pos_world + cam_extr[:3, 3]

                X_prev, Y_prev, Z_prev = prev_pos_cam
                pos_mask = (X_obs > X_prev - r) & (X_obs < X_prev + r) & \
                           (Y_obs > Y_prev - r) & (Y_obs < Y_prev + r) & \
                           (Z_obs > Z_prev - r) & (Z_obs < Z_prev + r)
                self.G.nodes[node]['obs_pts'] = pts_obs[pos_mask]

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
        camera_node = pyrender.Node(name='cam', camera=camera, matrix=np.eye(4))
        light_node = pyrender.Node(name='light', light=light, matrix=np.eye(4))
        scene = pyrender.Scene(nodes=[camera_node, light_node])

        # convert to anti-human opengl coordinates
        m = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        color_map = {'red': 1, 'green': 2, 'blue': 3}
        seg_node_map = dict()

        max_iter=30
        max_correspondence_distance = 0.03

        tic = time.time()
        for iter in range(max_iter):
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                init_pose = cam_extr @ self.G.edges[u, v]['pose_list'][-1]  # in camera frame
                mesh_node = pyrender.Node(name=f'{color}-mesh', mesh=self.rod_mesh, matrix=m @ init_pose)
                scene.add_node(mesh_node)
                seg_node_map[mesh_node] = color_map[color]

            seg_im, rendered_depth_im = self.renderer.render(scene, flags=RenderFlags.SEG, seg_node_map=seg_node_map)
            seg_im = torch.from_numpy(np.ascontiguousarray(seg_im[:, :, 0])).cuda()
            rendered_depth_im = torch.from_numpy(rendered_depth_im).cuda()

            node_to_pq = dict()
            for idx, (color, (u, v)) in enumerate(self.data_cfg['color_to_rod'].items()):
                indices = torch.nonzero((rendered_depth_im > 0) & (seg_im == color_map[color]))
                r_ys = indices[:, 0]
                r_xs = indices[:, 1]
                Z_est = rendered_depth_im[r_ys, r_xs]
                X_est = (r_xs - cx) / fx * Z_est
                Y_est = (r_ys - cy) / fy * Z_est
                rod_pts_est = torch.stack([X_est, Y_est, Z_est]).T  # (M, 3)

                for node in (u, v):
                    prev_pos_world = self.G.nodes[node]['pos_list'][-1]
                    prev_pos_cam = cam_extr[:3, :3] @ prev_pos_world + cam_extr[:3, 3]

                    X_prev, Y_prev, Z_prev = prev_pos_cam
                    node_pos_mask = (X_est > X_prev - r) & (X_est < X_prev + r) & \
                                    (Y_est > Y_prev - r) & (Y_est < Y_prev + r) & \
                                    (Z_est > Z_prev - r) & (Z_est < Z_prev + r)
                    node_pts_est = rod_pts_est[node_pos_mask]
                    node_pts_obs = self.G.nodes[node]['obs_pts']

                    distances = torch.norm(node_pts_obs.unsqueeze(1) - node_pts_est.unsqueeze(0), dim=2)  # (N, M)
                    closest_distance, closest_indices = torch.min(distances, axis=1)
                    dist_mask = closest_distance < max_correspondence_distance

                    Q = node_pts_obs[dist_mask]  # (K, 3)
                    P = node_pts_est[closest_indices][dist_mask]  # (K, 3)

                    node_to_pq[node] = (P, Q)

            # optimize use SGD
            num_rod = len(self.data_cfg['color_to_rod'])
            delta_T = torch.zeros((num_rod, 7), device='cuda')  # [[x, y, z, wx, wy, wz, ww]]
            delta_T[:, 6] = 1
            delta_T.requires_grad = True
            delta_T = SE3.InitFromVec(delta_T)
            delta_T = LieGroupParameter(delta_T)
            optimizer = SGD([delta_T], lr=0.001, momentum=0.9)

            n_steps = 10
            for _ in range(n_steps):
                loss = 0
                optimizer.zero_grad()
                for idx, (color, (u, v)) in enumerate(self.data_cfg['color_to_rod'].items()):
                    for node in (u, v):
                        P, Q = node_to_pq[node]
                        loss += torch.mean(torch.norm(Q - delta_T[idx, None] * P, dim=1))
                loss.backward()
                optimizer.step()

            break

        exit(0)


    def projective_icp_cuda(self, depth_im, mesh_nodes, init_poses, max_iter=30, max_distance=0.02,
                            early_stop_thresh=0.0015, verbose=False):
        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
        camera_node = pyrender.Node(name='cam', camera=camera, matrix=np.eye(4))
        light_node = pyrender.Node(name='light', light=light, matrix=np.eye(4))
        scene = pyrender.Scene(nodes=[camera_node, light_node])
        for mesh_node in mesh_nodes:
            scene.add_node(mesh_node)

        m = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        cam_extr = np.asarray(self.data_cfg['cam_extr'])
        T_list = [cam_extr @ init_pose for init_pose in init_poses] # convert to cam frame

        depth_im = torch.from_numpy(depth_im).cuda()
        indices = torch.nonzero(depth_im > 0)
        ys = indices[:, 0]
        xs = indices[:, 1]
        Z_obs = depth_im[ys, xs]
        X_obs = (xs - cx) / fx * Z_obs
        Y_obs = (ys - cy) / fy * Z_obs
        pts_obs = torch.stack([X_obs, Y_obs, Z_obs]).T  # (N, 3)

        for _ in range(max_iter):
            for mesh_node, T in zip(mesh_nodes, T_list):
                scene.set_pose(mesh_node, m @ T)
            _, rendered_depth_im = self.renderer.render(scene)
            rendered_depth_im = torch.from_numpy(rendered_depth_im).cuda()

            indices = np.nonzero(rendered_depth_im > 0)
            r_ys = indices[:, 0]
            r_xs = indices[:, 1]
            Z_est = rendered_depth_im[r_ys, r_xs]
            X_est = (r_xs - cx) / fx * Z_est
            Y_est = (r_ys - cy) / fy * Z_est
            pts_est = torch.stack([X_est, Y_est, Z_est]).T  # (M, 3)

            # nearest neighbor data association
            distances = torch.norm(pts_obs.unsqueeze(1) - pts_est.unsqueeze(0), dim=2)  # (N, M)
            closest_distance, closest_indices = torch.min(distances, axis=1)
            dist_mask = closest_distance < max_distance

            # ICP ref: https://www.youtube.com/watch?v=djnd502836w&t=781s
            Q = pts_obs[dist_mask].T  # (3, K)
            P = pts_est[closest_indices][dist_mask].T  # (3, K)

            P_mean = P.mean(1, keepdims=True)  # (3, 1)
            Q_mean = Q.mean(1, keepdims=True)  # (3, 1)

            error_before = torch.mean(torch.norm(Q - P, dim=0)).item()

            H = (Q - Q_mean) @ (P - P_mean).T
            U, D, V = torch.svd(H)
            R = U @ V.T
            t = Q_mean - R @ P_mean

            error_after = torch.mean(torch.norm(Q - (R @ P + t), dim=0)).item()
            if verbose:
                print(f"{error_before = :.4f}, {error_after = :.4f}, delta = {error_before - error_after:.4f}")

            if error_after > error_before:
                break

            delta_T = np.eye(4)
            delta_T[:3, :3] = R.cpu().numpy()
            delta_T[:3, 3:4] = t.cpu().numpy()

            for i in range(len(T_list)):
                T_list[i] = delta_T @ T_list[i]

            if error_after < early_stop_thresh:
                break

        return [la.inv(cam_extr) @ T for T in T_list]


def visualize(data_cfg, pcd, visualizer):
    visualizer.clear_geometries()
    visualizer.add_geometry(pcd)

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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    # parser.add_argument("--video_id", default="fabric2")
    # parser.add_argument("--video_id", default="six_cameras10")
    parser.add_argument("--video_id", default="crawling_sim")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/untethered_rod_w_end_cap.ply")
    parser.add_argument("--rod_pcd_file", default="pcd/yale/untethered_rod_w_end_cap.pcd")
    parser.add_argument("--first_frame_id", default=0, type=int)
    parser.add_argument("-v", "--visualize", default=True, action="store_true")
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video_id)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'color'))])

    # read data config
    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    # read rod pcd
    assert os.path.isfile(args.rod_mesh_file) or os.path.isfile(args.rod_pcd_file), "rod geometry is not found!"
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

    # read rod triangle mesh
    rod_fuze_mesh = trimesh.load(args.rod_mesh_file)
    rod_fuze_mesh.apply_translation(-rod_fuze_mesh.centroid)
    rod_fuze_mesh.apply_scale(1 / data_cfg['rod_scale'])
    rod_mesh = pyrender.Mesh.from_trimesh(rod_fuze_mesh)
    tracker = Tracker(data_cfg, rod_pcd, rod_mesh)

    # initialize tracker with the first frame
    color_path = os.path.join(video_path, 'color', f'{prefixes[args.first_frame_id]}.png')
    depth_path = os.path.join(video_path, 'depth', f'{prefixes[args.first_frame_id]}.png')
    info_path = os.path.join(video_path, 'data', f'{prefixes[args.first_frame_id]}.json')
    color_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
    with open(info_path, 'r') as f:
        info = json.load(f)

    info['sensor_status'] = {key: True for key in info['sensors'].keys()}
    # info['sensor_status']['2'] = False

    tracker.initialize(color_im, depth_im, info, visualize=args.visualize, compute_hsv=False)
    data_cfg_module.write_config(tracker.data_cfg)

    # track frames
    os.makedirs(os.path.join(video_path, "scene_cloud"), exist_ok=True)
    os.makedirs(os.path.join(video_path, "estimation_cloud"), exist_ok=True)
    os.makedirs(os.path.join(video_path, "raw_estimation"), exist_ok=True)
    os.makedirs(os.path.join(video_path, "observation"), exist_ok=True)

    if args.visualize:
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=data_cfg['im_w'], height=data_cfg['im_h'])

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

        info['sensor_status'] = {key: True for key in info['sensors'].keys()}
        # info['sensor_status']['2'] = False

        complete_obs_color, complete_obs_depth = tracker.update(color_im, depth_im, info)
        # complete_obs_bgr = cv2.cvtColor(complete_obs_color, cv2.COLOR_RGB2BGR)
        # complete_obs_bgr[complete_obs_depth == 0] = 0
        # cv2.imwrite(os.path.join(video_path, 'observation', f"{idx:04d}.png"), complete_obs_bgr)

        if args.visualize:
            cv2.imshow("observation", color_im_bgr)
            cv2.waitKey(1)

            scene_pcd = perception_utils.create_pcd(depth_im, data_cfg['cam_intr'], color_im,
                                                    depth_trunc=data_cfg['depth_trunc'],
                                                    cam_extr=data_cfg['cam_extr'])

            robot_cloud = o3d.geometry.PointCloud()
            for color, (u, v) in data_cfg['color_to_rod'].items():
                rod_pcd = copy.deepcopy(tracker.rod_pcd)
                rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                rod_pose = tracker.G.edges[u, v]['pose_list'][-1]
                rod_pcd.transform(rod_pose)
                robot_cloud += rod_pcd

            estimation_cloud = robot_cloud + scene_pcd
            visualize(data_cfg, estimation_cloud, visualizer)
            # visualizer.capture_screen_image(os.path.join(video_path, "raw_estimation", f"{idx:04d}.png"))
            # o3d.io.write_point_cloud(os.path.join(video_path, "estimation_cloud", f"{idx:04d}.ply"), estimation_cloud)

    # save rod poses and end cap positions to file
    pose_output_folder = os.path.join(video_path, "poses")
    os.makedirs(pose_output_folder, exist_ok=True)
    for color, (u, v) in tracker.data_cfg['color_to_rod'].items():
        np.save(os.path.join(pose_output_folder, f'{color}.npy'), np.array(tracker.G.edges[u, v]['pose_list']))
        np.save(os.path.join(pose_output_folder, f'{u}_pos.npy'), np.array(tracker.G.nodes[u]['pos_list']))
        np.save(os.path.join(pose_output_folder, f'{v}_pos.npy'), np.array(tracker.G.nodes[v]['pos_list']))
