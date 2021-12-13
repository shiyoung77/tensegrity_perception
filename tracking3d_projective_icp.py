import os
import time
import copy
import importlib
import pprint
import json
from argparse import ArgumentParser

import numpy as np
import numpy.linalg as la
import torch
import cv2
import networkx as nx
import open3d as o3d
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm
import trimesh
import pyrender
from pyrender.constants import RenderFlags

import perception_utils


class Tracker:
    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }

    def __init__(self, data_cfg, rod_pcd=None, end_cap_meshes=None):
        self.data_cfg = data_cfg
        pprint.pprint(self.data_cfg)
        self.initialized = False

        self.end_cap_meshes = end_cap_meshes
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
                    hsv_mask |= cv2.inRange(color_im_hsv, lowerb=(0, 50, 50), upperb=(20, 255, 255)).astype(np.bool8)
                mask &= hsv_mask
                if mask.sum() == 0:
                    print(color, ' has not points at node ', node)
                    assert mask.sum() > 0
                else:
                    print(f"{color} node id {node} has {mask.sum()} points in pcd.")
                obs_depth_im[mask] = depth_im[mask]
                masked_depth_im = depth_im * mask
                end_cap_pcd = perception_utils.create_pcd(masked_depth_im, self.data_cfg['cam_intr'],
                                                          cam_extr=self.data_cfg['cam_extr'])

                # # filter end_cap_pcd by finding the largest cluster
                # labels = np.asarray(end_cap_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
                # points_for_each_cluster = [(labels == label).sum() for label in range(labels.max() + 1)]
                # label = np.argmax(points_for_each_cluster)
                # masked_indices = np.where(labels == label)[0]
                # end_cap_pcd = end_cap_pcd.select_by_index(masked_indices)
                # o3d.visualization.draw_geometries([end_cap_pcd])

                end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)
                end_cap_centers.append(end_cap_center)

                complete_obs_mask |= mask

            # compute rod pose given end cap centers
            init_pose = self.estimate_rod_pose_from_end_cap_centers(end_cap_centers)
            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)

            # icp refinement
            mesh_nodes = [pyrender.Node(name='mesh', mesh=self.end_cap_meshes[0], matrix=np.eye(4)),
                          pyrender.Node(name='mesh', mesh=self.end_cap_meshes[1], matrix=np.eye(4))]
            init_rod_pose_cam = self.data_cfg['cam_extr'] @ init_pose
            init_poses = [init_rod_pose_cam, init_rod_pose_cam]
            rod_pose, rendered_depth, rendered_seg = self.projective_icp_cuda(
                obs_depth_im, mesh_nodes, init_poses, max_iter=30, max_distance=0.02, early_stop_thresh=0.0015)
            rod_pose = la.inv(self.data_cfg['cam_extr']) @ rod_pose[0]

            self.G.edges[u, v]['pose_list'].append(rod_pose)
            self.G.edges[u, v]['rendered_depth'] = rendered_depth
            self.G.nodes[u]['pos_list'].append(rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[v]['pos_list'].append(rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[u]['pos_2d'], self.G.nodes[u]['roi_2d'] = self.get_end_cap_roi(u, radius=0.03)
            self.G.nodes[v]['pos_2d'], self.G.nodes[v]['roi_2d'] = self.get_end_cap_roi(v, radius=0.03)

        """
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            rendered_depth = self.G.edges[u, v]['rendered_depth']
            vis = np.zeros_like(color_im)
            vis[rendered_depth > 0] = [255, 255, 255]

            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color=(0, 0, 255))

            x_min, y_min, x_max, y_max = self.get_end_cap_roi(v, radius=0.03)
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color=(0, 0, 255))

            plt.imshow(vis)
            plt.show()
        # exit(0)
        """

        complete_obs_color[complete_obs_mask] = color_im[complete_obs_mask]
        complete_obs_depth[complete_obs_mask] = depth_im[complete_obs_mask]

        self.constrained_optimization(info)
        # self.rigid_finetune(complete_obs_depth)

        if visualize:
            complete_obs_hsv = cv2.cvtColor(complete_obs_color, cv2.COLOR_RGB2HSV)
            _, axes = plt.subplots(2, 2)
            axes[0, 0].imshow(color_im)
            axes[0, 1].imshow(complete_obs_color)
            axes[1, 0].imshow(complete_obs_hsv)
            axes[1, 1].imshow(complete_obs_depth)
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

    def update(self, color_im, depth_im, info, max_iter=3):
        assert self.initialized, "[Error] You must first initialize the tracker!"

        color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)
        H, W, _ = color_im_hsv.shape

        for iter in range(max_iter):
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                prev_pose = self.G.edges[u, v]['pose_list'][-1].copy()
                obs_depth_im = np.zeros_like(depth_im)

                hsv_mask = cv2.inRange(color_im_hsv,
                                    lowerb=tuple(self.data_cfg['hsv_ranges'][color][0]),
                                    upperb=tuple(self.data_cfg['hsv_ranges'][color][1])).astype(np.bool8)
                if color == 'red':
                    hsv_mask |= cv2.inRange(color_im_hsv, lowerb=(0, 80, 50), upperb=(15, 255, 255)).astype(np.bool8)

                obs_depth_im[hsv_mask] = depth_im[hsv_mask]
                aug_obs_depth_im = obs_depth_im.copy()

                vis = np.zeros_like(color_im)
                vis[obs_depth_im > 0] = 255
                for node in (u, v):
                    center_x, center_y = self.G.nodes[node]['pos_2d']
                    x_min, x_max = center_x - 40, center_x + 40
                    y_min, y_max = center_y - 40, center_y + 40
                    cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color=(255, 0, 0))
                    cv2.putText(vis, str(node), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

                    ys, xs = np.nonzero(hsv_mask[y_min:y_max, x_min:x_max])
                    num_obs_pixels = ys.shape[0]
                    self.G.nodes[node]['num_obs_pixels'] = num_obs_pixels

                    if num_obs_pixels < 50:
                        center_x, center_y = self.G.nodes[node]['pos_2d']
                        x_min, x_max = center_x - 15, center_x + 15
                        y_min, y_max = center_y - 15, center_y + 15
                        prev_roi_depth = self.G.edges[u, v]['rendered_depth'][y_min:y_max, x_min:x_max]
                        aug_obs_depth_im[y_min:y_max, x_min:x_max] = prev_roi_depth
                        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color=(0, 0, 255))

                # ICP
                mesh_nodes = [pyrender.Node(name='mesh', mesh=self.end_cap_meshes[0], matrix=np.eye(4)),
                            pyrender.Node(name='mesh', mesh=self.end_cap_meshes[1], matrix=np.eye(4))]
                prev_pose_cam = self.data_cfg['cam_extr'] @ prev_pose
                init_poses = [prev_pose_cam, prev_pose_cam]
                # tic = time.time()
                rod_pose, rendered_depth, rendered_seg = self.projective_icp_cuda(
                    aug_obs_depth_im, mesh_nodes, init_poses, max_iter=1, max_distance=0.1, early_stop_thresh=0.001)
                # print(f"icp takes {time.time() - tic}s")
                rod_pose = la.inv(self.data_cfg['cam_extr']) @ rod_pose[0]
                confidence_u, confidence_v = self.compute_confidence(obs_depth_im, rendered_depth, rendered_seg,
                                                                    inlier_thresh=0.03)

                if iter == 0:
                    self.G.edges[u, v]['pose_list'].append(rod_pose)
                    self.G.nodes[u]['pos_list'].append(rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2)
                    self.G.nodes[v]['pos_list'].append(rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2)
                else:
                    self.G.edges[u, v]['pose_list'][-1] = rod_pose
                    self.G.nodes[u]['pos_list'][-1] = rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2
                    self.G.nodes[v]['pos_list'][-1] = rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2

                self.G.edges[u, v]['rendered_depth'] = rendered_depth
                self.G.nodes[u]['pos_2d'], self.G.nodes[u]['roi_2d'] = self.get_end_cap_roi(u, radius=0.03)
                self.G.nodes[v]['pos_2d'], self.G.nodes[v]['roi_2d'] = self.get_end_cap_roi(v, radius=0.03)
                self.G.nodes[u]['confidence'] = confidence_u
                self.G.nodes[v]['confidence'] = confidence_v

                if iter == max_iter - 1:
                    if color == 'green':
                        vis[rendered_depth > 0] = [0, 255, 0]
                        cv2.imshow("green_obs", vis)
                    elif color == 'blue':
                        vis[rendered_depth > 0] = [255, 0, 0]
                        cv2.imshow("blue_obs", vis)
                    elif color == 'red':
                        vis[rendered_depth > 0] = [0, 0, 255]
                        cv2.imshow("red_obs", vis)

            # tic = time.time()
            self.constrained_optimization(info)
            # print(f"optimization takes {time.time() - tic}s")

        return

    def compute_mc2cam_mat(self, info):
        cap_end_pos = list()
        num_end_caps = 2 * self.data_cfg['num_rods']
        for i in range(num_end_caps):
            cap_end_pos.append(np.concatenate([self.G.nodes[i]['pos_list'][-1], [1.]]))
        extr_mat = self.data_cfg['cam_extr']
        cap_end_pos_in_cam_f = np.matmul(extr_mat, np.array(cap_end_pos).T)  # 4x6
        cap_end_pos_in_mc_f = list()
        mask_ids = list()
        for i in range(num_end_caps):
            pos = info['mocap'][str(i)]
            if pos is None:
                continue
            mask_ids.append(i)
            cap_end_pos_in_mc_f.append([pos['x']/1000., pos['y']/1000., pos['z']/1000.])
        cap_end_pos_in_mc_f = np.array(cap_end_pos_in_mc_f).T  # 3x6
        cap_end_pos_in_cam_f = cap_end_pos_in_cam_f[:, mask_ids]

        P = cap_end_pos_in_cam_f[:3]
        Q = cap_end_pos_in_mc_f
        P_mean = P.mean(1, keepdims=True)  # (3, 1)
        Q_mean = Q.mean(1, keepdims=True)  # (3, 1)

        H = (Q - Q_mean) @ (P - P_mean).T
        U, D, V = la.svd(H)
        R = U @ V.T
        t = Q_mean - R @ P_mean

        cam2mc_mat = np.zeros((4, 4))
        cam2mc_mat[0:3, 0:3] = R
        cam2mc_mat[0:3, 3] = t.squeeze()
        print(cam2mc_mat)
        return cam2mc_mat

    def compute_confidence(self, obs_depth_im, rendered_depth_im, rendered_seg_im, inlier_thresh=0.01):
        seg_gpu = torch.from_numpy(rendered_seg_im).cuda()
        obs_depth_gpu = torch.from_numpy(obs_depth_im).cuda()
        rendered_depth_gpu = torch.from_numpy(rendered_depth_im).cuda()

        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        indices = torch.nonzero(obs_depth_gpu > 0)
        ys, xs = indices[:, 0], indices[:, 1]
        Z_obs = obs_depth_gpu[ys, xs]
        X_obs = (xs - cx) / fx * Z_obs
        Y_obs = (ys - cy) / fy * Z_obs
        pts_obs = torch.stack([X_obs, Y_obs, Z_obs]).T  # (N, 3)

        indices = np.nonzero(rendered_depth_gpu > 0)
        r_ys = indices[:, 0]
        r_xs = indices[:, 1]
        Z_est = rendered_depth_gpu[r_ys, r_xs]
        X_est = (r_xs - cx) / fx * Z_est
        Y_est = (r_ys - cy) / fy * Z_est
        pts_est = torch.stack([X_est, Y_est, Z_est]).T  # (M, 3)

        pts_labels = seg_gpu[r_ys, r_xs]
        rendered_u = torch.sum(pts_labels == 1).cpu().item()
        rendered_v = torch.sum(pts_labels == 2).cpu().item()

        # nearest neighbor data association
        distances = torch.norm(pts_est.unsqueeze(1) - pts_obs.unsqueeze(0), dim=2)  # (N, M)
        closest_distance, closest_indices = torch.min(distances, dim=1)
        dist_mask = closest_distance < inlier_thresh

        # get inlier labels
        matched_labels = pts_labels[dist_mask]
        matched_u = torch.sum(matched_labels == 1).cpu().item()
        matched_v = torch.sum(matched_labels == 2).cpu().item()
        return max(0.1, matched_u / rendered_u), max(0.1, matched_v / rendered_v)

    def constrained_optimization(self, info):
        rod_length = self.data_cfg['rod_length']
        num_end_caps = 2 * self.data_cfg['num_rods']  # a rod has two end caps

        sensor_measurement = info['sensors']
        sensor_status = info['sensor_status']
        obj_func = self.objective_function_generator(sensor_measurement, sensor_status)

        init_values = np.zeros(3 * num_end_caps)
        for i in range(num_end_caps):
            init_values[(3 * i):(3 * i + 3)] = self.G.nodes[i]['pos_list'][-1]

        # be VERY CAREFUL when generating a lambda function in a for loop
        # https://stackoverflow.com/questions/45491376/
        # https://docs.python-guide.org/writing/gotchas/#late-binding-closures
        rod_constraints = []
        for _, (u, v) in self.data_cfg['color_to_rod'].items():
            constraint = dict()
            constraint['type'] = 'eq'
            constraint['fun'] = self.constraint_function_generator(u, v, rod_length)
            rod_constraints.append(constraint)

        res = minimize(obj_func, init_values, method='SLSQP', constraints=rod_constraints)
        assert res.success, "Optimization fail! Something must be wrong."

        for i in range(num_end_caps):
            self.G.nodes[i]['pos_list'][-1] = res.x[(3 * i):(3 * i + 3)].copy()

        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            prev_rod_pose = self.G.edges[u, v]['pose_list'][-1]
            u_pos = self.G.nodes[u]['pos_list'][-1]
            v_pos = self.G.nodes[v]['pos_list'][-1]
            curr_end_cap_centers = np.vstack([u_pos, v_pos])
            optimized_pose = self.estimate_rod_pose_from_end_cap_centers(curr_end_cap_centers, prev_rod_pose)
            self.G.edges[u, v]['pose_list'][-1] = optimized_pose

    def objective_function_generator(self, sensor_measurement, sensor_status):

        def objective_function(X):
            unary_loss = 0
            for i in range(len(self.data_cfg['node_to_color'])):
                pos = X[(3 * i):(3 * i + 3)]
                estimated_pos = self.G.nodes[i]['pos_list'][-1]
                confidence = self.G.nodes[i].get('confidence', 1)
                unary_loss += confidence * np.sum((pos - estimated_pos)**2)

            binary_loss = 0
            for sensor_id, (u, v) in self.data_cfg['sensor_to_tendon'].items():
                sensor_id = str(sensor_id)
                if not sensor_status[sensor_id] or sensor_measurement[sensor_id]['length'] <= 0:
                    continue

                u_confidence = self.G.nodes[u].get('confidence', 1)
                v_confidence = self.G.nodes[v].get('confidence', 1)
                factor = 0.5
                if u_confidence > 0.5 and v_confidence > 0.5:
                    factor = 0
                if u_confidence < 0.2 or v_confidence <= 0.2:
                    factor = 1

                u_pos = X[(3 * u):(3 * u + 3)]
                v_pos = X[(3 * v):(3 * v + 3)]
                estimated_length = la.norm(u_pos - v_pos)
                measured_length = sensor_measurement[sensor_id]['length'] / 100
                binary_loss += factor * (estimated_length - measured_length)**2

            return unary_loss + binary_loss

        return objective_function

    def constraint_function_generator(self, u, v, rod_length):

        def constraint_function(X):
            u_pos = X[3 * u: 3 * u + 3]
            v_pos = X[3 * v: 3 * v + 3]
            return la.norm(u_pos - v_pos) - rod_length

        return constraint_function

    def rigid_finetune(self, complete_obs_depth):
        mesh_nodes = []
        init_poses = []
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            mesh_node = pyrender.Node(name=f'{color}-mesh', mesh=self.rod_mesh, matrix=np.eye(4))
            prev_rod_pose = self.G.edges[u, v]['pose_list'][-1]
            mesh_nodes.append(mesh_node)
            init_poses.append(prev_rod_pose)

        finetuned_poses, _ = self.projective_icp_cuda(complete_obs_depth, mesh_nodes, init_poses, max_iter=30,
                                                      max_distance=0.02, early_stop_thresh=0.0015, verbose=False)

        for i, (u, v) in enumerate(self.data_cfg['color_to_rod'].values()):
            finetuned_pose = finetuned_poses[i]
            self.G.edges[u, v]['pose_list'][-1] = finetuned_pose
            self.G.nodes[u]['pos_list'][-1] = finetuned_pose[:3, 3] + finetuned_pose[:3, 2] * self.rod_length / 2
            self.G.nodes[v]['pos_list'][-1] = finetuned_pose[:3, 3] - finetuned_pose[:3, 2] * self.rod_length / 2

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

    def get_end_cap_roi(self, end_cap_id, radius=0.03):
        pos_world = self.G.nodes[end_cap_id]['pos_list'][-1]
        X, Y, Z = pos_world
        pos_cam = self.data_cfg['cam_extr'] @ np.array([X, Y, Z, 1])
        X, Y, Z = pos_cam[:3]

        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        corner_pts = np.array([
            [X, Y, Z, 1],
            [X - radius, Y - radius, Z - radius, 1],
            [X - radius, Y - radius, Z + radius, 1],
            [X + radius, Y - radius, Z - radius, 1],
            [X + radius, Y - radius, Z + radius, 1],
            [X - radius, Y + radius, Z - radius, 1],
            [X - radius, Y + radius, Z + radius, 1],
            [X + radius, Y + radius, Z - radius, 1],
            [X + radius, Y + radius, Z + radius, 1],
        ])

        xs = (fx * corner_pts[:, 0] / corner_pts[:, 2] + cx).astype(np.int32)
        ys = (fy * corner_pts[:, 1] / corner_pts[:, 2] + cy).astype(np.int32)
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        return (xs[0], ys[0]), (x_min, y_min, x_max, y_max)

    def projective_icp_cuda(self, depth_im, mesh_nodes, init_poses, max_iter=30, max_distance=0.02,
                            early_stop_thresh=0.0015, verbose=False):
        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0)
        camera_node = pyrender.Node(name='cam', camera=camera, matrix=np.eye(4))
        light_node = pyrender.Node(name='light', light=light, matrix=np.eye(4))
        scene = pyrender.Scene(nodes=[camera_node, light_node])
        seg_node_map = dict()
        for i, mesh_node in enumerate(mesh_nodes):
            scene.add_node(mesh_node)
            seg_node_map[mesh_node] = i + 1

        m = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        depth_im = torch.from_numpy(depth_im).cuda()
        indices = torch.nonzero(depth_im > 0)
        ys = indices[:, 0]
        xs = indices[:, 1]
        Z_obs = depth_im[ys, xs]
        X_obs = (xs - cx) / fx * Z_obs
        Y_obs = (ys - cy) / fy * Z_obs
        pts_obs = torch.stack([X_obs, Y_obs, Z_obs]).T  # (N, 3)

        T_list = [pose.copy() for pose in init_poses]

        for _ in range(max_iter):
            for mesh_node, T in zip(mesh_nodes, T_list):
                scene.set_pose(mesh_node, m @ T)

            rendered_depth_im = self.renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
            rendered_depth_im_gpu = torch.from_numpy(rendered_depth_im).cuda()

            indices = np.nonzero(rendered_depth_im_gpu > 0)
            r_ys = indices[:, 0]
            r_xs = indices[:, 1]
            Z_est = rendered_depth_im_gpu[r_ys, r_xs]
            X_est = (r_xs - cx) / fx * Z_est
            Y_est = (r_ys - cy) / fy * Z_est
            pts_est = torch.stack([X_est, Y_est, Z_est]).T  # (M, 3)

            # nearest neighbor data association
            distances = torch.norm(pts_obs.unsqueeze(1) - pts_est.unsqueeze(0), dim=2)  # (N, M)
            closest_distance, closest_indices = torch.min(distances, dim=1)
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

        # render final depth and segmentation image
        for mesh_node, T in zip(mesh_nodes, T_list):
            scene.set_pose(mesh_node, m @ T)
        rendered_seg, rendered_depth = self.renderer.render(scene, flags=RenderFlags.SEG, seg_node_map=seg_node_map)
        rendered_seg = rendered_seg[:, :, 0].copy()

        return T_list, rendered_depth, rendered_seg


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
    # parser.add_argument("--video_id", default="crawling_sim")
    parser.add_argument("--video_id", default="socks6")
    # parser.add_argument("--rod_mesh_file", default="pcd/yale/untethered_rod_w_end_cap.ply")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/end_cap_only_new.obj")
    parser.add_argument("--top_end_cap_mesh_file", default="pcd/yale/end_cap_top.obj")
    parser.add_argument("--bottom_end_cap_mesh_file", default="pcd/yale/end_cap_bottom.obj")
    parser.add_argument("--rod_pcd_file", default="pcd/yale/untethered_rod_w_end_cap.pcd")
    parser.add_argument("--first_frame_id", default=50, type=int)
    parser.add_argument("--max_iter", default=3, type=int)
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
        rod_pcd = rod_mesh.sample_points_poisson_disk(3000)
        points = np.asarray(rod_pcd.points)
        offset = points.mean(0)
        points -= offset  # move point cloud center
        points /= data_cfg['rod_scale']  # scale points from millimeter to meter
        pcd_path = ".".join(args.rod_mesh_file.split('.')[:-1]) + ".pcd"
        o3d.io.write_point_cloud(pcd_path, rod_pcd)
        print(f"rod pcd file is generated and saved to {pcd_path}")

    # read rod triangle mesh
    # rod_fuze_mesh = trimesh.load_mesh(args.rod_mesh_file)
    # print(rod_fuze_mesh.centroid)
    # rod_fuze_mesh.apply_translation(-rod_fuze_mesh.centroid)
    # rod_fuze_mesh.apply_scale(1 / data_cfg['rod_scale'])
    # rod_mesh = pyrender.Mesh.from_trimesh(rod_fuze_mesh)

    top_end_cap_mesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(args.top_end_cap_mesh_file))
    bottom_end_cap_mesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(args.bottom_end_cap_mesh_file))
    tracker = Tracker(data_cfg, rod_pcd, [top_end_cap_mesh, bottom_end_cap_mesh])

    # initialize tracker with the first frame
    color_path = os.path.join(video_path, 'color', f'{prefixes[args.first_frame_id]}.png')
    depth_path = os.path.join(video_path, 'depth', f'{prefixes[args.first_frame_id]}.png')
    info_path = os.path.join(video_path, 'data', f'{prefixes[args.first_frame_id]}.json')
    color_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
    with open(info_path, 'r') as f:
        info = json.load(f)

    info['sensor_status'] = {key: True for key in info['sensors'].keys()}
    info['sensor_status']['6'] = False
    info['sensor_status']['7'] = False
    info['sensor_status']['8'] = False

    tracker.initialize(color_im, depth_im, info, visualize=args.visualize, compute_hsv=False)
    data_cfg_module.write_config(tracker.data_cfg)

    # track frames
    os.makedirs(os.path.join(video_path, "scene_cloud"), exist_ok=True)
    os.makedirs(os.path.join(video_path, "estimation_cloud"), exist_ok=True)
    os.makedirs(os.path.join(video_path, "raw_estimation"), exist_ok=True)

    if args.visualize:
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=data_cfg['im_w'], height=data_cfg['im_h'])
        cv2.namedWindow("observation")
        cv2.moveWindow("observation", 800, 100)
        cv2.namedWindow("red_obs")
        cv2.moveWindow("red_obs", 100, 600)
        cv2.namedWindow("blue_obs")
        cv2.moveWindow("blue_obs", 800, 600)
        cv2.namedWindow("green_obs")
        cv2.moveWindow("green_obs", 1500, 600)

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
        info['sensor_status']['6'] = False
        info['sensor_status']['7'] = False
        info['sensor_status']['8'] = False

        tracker.update(color_im, depth_im, info, args.max_iter)

        if args.visualize:
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

            cv2.imshow("observation", color_im_bgr)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit(0)

    # save rod poses and end cap positions to file
    pose_output_folder = os.path.join(video_path, "poses")
    os.makedirs(pose_output_folder, exist_ok=True)
    for color, (u, v) in tracker.data_cfg['color_to_rod'].items():
        np.save(os.path.join(pose_output_folder, f'{color}.npy'), np.array(tracker.G.edges[u, v]['pose_list']))
        np.save(os.path.join(pose_output_folder, f'{u}_pos.npy'), np.array(tracker.G.nodes[u]['pos_list']))
        np.save(os.path.join(pose_output_folder, f'{v}_pos.npy'), np.array(tracker.G.nodes[v]['pos_list']))
