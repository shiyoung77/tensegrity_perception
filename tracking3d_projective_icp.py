import os
import time
import copy
import importlib
import pprint
import json
from argparse import ArgumentParser
from typing import List, Dict

import numpy as np
import scipy.linalg as la
import torch
import cv2
import networkx as nx
import open3d as o3d
from pyquaternion import Quaternion
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

    def __init__(self, data_cfg, rod_pcd=None, rod_mesh=None, end_cap_meshes=None):
        self.data_cfg = data_cfg
        pprint.pprint(self.data_cfg)
        self.initialized = False

        self.end_cap_meshes = end_cap_meshes
        self.rod_pcd = copy.deepcopy(rod_pcd)
        self.rod_mesh = rod_mesh
        self.rod_length = self.data_cfg['rod_length']

        # set up pyrender scene
        H, W = self.data_cfg['im_h'], self.data_cfg['im_w']
        self.renderer = pyrender.OffscreenRenderer(W, H)
        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx = cam_intr[0, 0]
        fy = cam_intr[1, 1]
        cx = cam_intr[0, 2]
        cy = cam_intr[1, 2]
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0)
        camera_node = pyrender.Node(name='cam', camera=camera, matrix=np.eye(4))
        light_node = pyrender.Node(name='light', light=light, matrix=np.eye(4))
        self.render_scene = pyrender.Scene(nodes=[camera_node, light_node])

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

        # vis_rendered_pts = []
        # vis_obs_pts = []

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
                    hsv_mask |= cv2.inRange(color_im_hsv, lowerb=(0, 80, 50), upperb=(20, 255, 255)).astype(np.bool8)
                mask &= hsv_mask
                assert mask.sum() > 0, f"node {node} ({color}) has no points observable in the initial frame."
                print(f"{color}-{u} has {mask.sum()} points.")
                obs_depth_im[mask] = depth_im[mask]
                masked_depth_im = depth_im * mask
                complete_obs_mask |= mask

                # filter observable end cap point cloud by finding the largest cluster
                end_cap_pcd = perception_utils.create_pcd(masked_depth_im, self.data_cfg['cam_intr'])
                labels = np.asarray(end_cap_pcd.cluster_dbscan(eps=0.005, min_points=10, print_progress=False))
                points_for_each_cluster = [(labels == label).sum() for label in range(labels.max() + 1)]
                label = np.argmax(points_for_each_cluster)
                masked_indices = np.where(labels == label)[0]
                end_cap_pcd = end_cap_pcd.select_by_index(masked_indices)
                end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)
                end_cap_centers.append(end_cap_center)

            # icp refinement
            obs_pts, _ = self.back_projection(torch.from_numpy(obs_depth_im).cuda())
            init_pose = self.estimate_rod_pose_from_end_cap_centers(end_cap_centers)
            mesh_nodes = [self.create_scene_node(str(i), self.end_cap_meshes[i], init_pose) for i in range(2)]
            rendered_depth_im = self.render_nodes(mesh_nodes, depth_only=True)
            rendered_pts, _ = self.back_projection(torch.from_numpy(rendered_depth_im).cuda())
            rod_pose, Q, P = self.projective_icp_cuda(obs_pts, rendered_pts, init_pose, max_distance=0.1, verbose=False)

            # vis_obs_pts.append(Q)
            # vis_rendered_pts.append(P)

            self.G.edges[u, v]['pose_list'].append(rod_pose)
            self.G.nodes[u]['pos_list'].append(rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[v]['pos_list'].append(rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2)

        # vis_obs_pts = torch.vstack(vis_obs_pts).cpu().numpy()
        # obs_pcd = o3d.geometry.PointCloud()
        # obs_pcd.points = o3d.utility.Vector3dVector(vis_obs_pts)
        # obs_pcd = obs_pcd.paint_uniform_color([0, 0, 0])
        # vis_rendered_pts = torch.vstack(vis_rendered_pts).cpu().numpy()
        # rendered_pcd = o3d.geometry.PointCloud()
        # rendered_pcd.points = o3d.utility.Vector3dVector(vis_rendered_pts)
        # o3d.visualization.draw_geometries([rendered_pcd, obs_pcd])

        complete_obs_color[complete_obs_mask] = color_im[complete_obs_mask]
        complete_obs_depth[complete_obs_mask] = depth_im[complete_obs_mask]

        self.constrained_optimization(info)
        # self.rigid_finetune(complete_obs_depth)

        if visualize:
            hsv_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)
            _, axes = plt.subplots(2, 2)
            axes[0, 0].imshow(color_im)
            axes[0, 1].imshow(complete_obs_color)
            axes[1, 0].imshow(hsv_im)
            axes[1, 1].imshow(complete_obs_depth)
            plt.show()

            vis_geometries = [scene_pcd]
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                rod_pcd = copy.deepcopy(self.rod_pcd)
                rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                rod_pose = self.G.edges[u, v]['pose_list'][-1]
                rod_pcd.transform(rod_pose)
                vis_geometries.append(rod_pcd)
            o3d.visualization.draw_geometries(vis_geometries)

        self.initialized = True


    def back_projection(self, depth_im: torch.Tensor):
        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx, fy = cam_intr[0, 0], cam_intr[1, 1]
        cx, cy = cam_intr[0, 2], cam_intr[1, 2]

        indices = torch.nonzero(depth_im > 0)
        ys, xs = indices[:, 0], indices[:, 1]
        Z_obs = depth_im[ys, xs]
        X_obs = (xs - cx) / fx * Z_obs
        Y_obs = (ys - cy) / fy * Z_obs
        pts = torch.stack([X_obs, Y_obs, Z_obs]).T  # (N, 3)
        return pts, (ys, xs)


    def compute_hsv_mask(self, color_im, color) -> np.ndarray:
        color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)
        hsv_mask = cv2.inRange(color_im_hsv,
                            lowerb=tuple(self.data_cfg['hsv_ranges'][color][0]),
                            upperb=tuple(self.data_cfg['hsv_ranges'][color][1])).astype(np.bool8)
        # if color == 'red':
        #     hsv_mask |= cv2.inRange(color_im_hsv, lowerb=(0, 80, 50), upperb=(15, 255, 255)).astype(np.bool8)
        return hsv_mask


    def compute_obs_pts(self, depth_im, mask):
        if isinstance(depth_im, np.ndarray):
            depth_im = torch.from_numpy(depth_im).cuda()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).cuda()

        obs_depth_im = torch.zeros_like(depth_im)
        obs_depth_im[mask] = depth_im[mask]
        obs_pts, _ = self.back_projection(obs_depth_im)
        return obs_pts


    def update(self, color_im, depth_im, info, max_iter=3):
        assert self.initialized, "[Error] You must first initialize the tracker!"

        H, W = depth_im.shape
        depth_im_gpu = torch.from_numpy(depth_im).cuda()

        # get observed 3D points for both end caps of each rod
        tic = time.time()
        obs_pts_dict = dict()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            hsv_mask = self.compute_hsv_mask(color_im, color)
            obs_pts = self.compute_obs_pts(depth_im_gpu, hsv_mask)

            # add fake points at the prev end cap position
            fake_pts = np.stack([self.G.nodes[u]['pos_list'][-1], self.G.nodes[v]['pos_list'][-1]])
            fake_pts = torch.from_numpy(fake_pts).to(torch.float32).cuda()
            fake_pts = torch.tile(fake_pts, (50, 1))

            augmented_obs_pts = torch.vstack([obs_pts, fake_pts])
            obs_pts_dict[color] = augmented_obs_pts
        # print(f"color filter takes: {time.time() - tic}s")

        for iter in range(max_iter):
            # render current state (end caps only)

            tic = time.time()
            mesh_nodes = dict()
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                prev_pose = self.G.edges[u, v]['pose_list'][-1].copy()
                mesh_nodes[self.create_scene_node(f"node-{u}", self.end_cap_meshes[0], prev_pose)] = u + 1
                mesh_nodes[self.create_scene_node(f"node-{v}", self.end_cap_meshes[1], prev_pose)] = v + 1

            rendered_seg, rendered_depth = self.render_nodes(mesh_nodes, depth_only=False)
            rendered_seg_gpu = torch.from_numpy(rendered_seg).cuda()
            rendered_depth_gpu = torch.from_numpy(rendered_depth).cuda()
            # print(f"rendering takes {time.time() - tic}s")

            # ================================== prediction step ==================================
            tic = time.time()
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                mask = torch.zeros((H, W), dtype=torch.bool, device='cuda:0')
                mask[rendered_seg_gpu == u + 1] = True
                mask[rendered_seg_gpu == v + 1] = True
                rendered_pts = self.compute_obs_pts(rendered_depth_gpu, mask)

                # add fake points at the prev end cap position
                fake_pts = np.stack([self.G.nodes[u]['pos_list'][-1], self.G.nodes[v]['pos_list'][-1]])
                fake_pts = torch.from_numpy(fake_pts).to(torch.float32).cuda()
                fake_pts = torch.tile(fake_pts, (50, 1))
                rendered_pts = torch.vstack([rendered_pts, fake_pts])

                obs_pts = obs_pts_dict[color]
                prev_pose = self.G.edges[u, v]['pose_list'][-1].copy()
                rod_pose, Q, P = self.projective_icp_cuda(obs_pts, rendered_pts, prev_pose, max_distance=0.1, verbose=False)

                if iter == 0:
                    self.G.edges[u, v]['pose_list'].append(rod_pose)
                    self.G.nodes[u]['pos_list'].append(rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2)
                    self.G.nodes[v]['pos_list'].append(rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2)
                else:
                    self.G.edges[u, v]['pose_list'][-1] = rod_pose
                    self.G.nodes[u]['pos_list'][-1] = rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2
                    self.G.nodes[v]['pos_list'][-1] = rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2
            # print(f"ICP takes {time.time() - tic}s")

            # ================================== correction step ==================================
            # render final depth and segmentation image
            # for mesh_node, T in zip(mesh_nodes, T_list):
            #     scene.set_pose(mesh_node, m @ T)
            # rendered_seg, rendered_depth = self.renderer.render(scene, flags=RenderFlags.SEG, seg_node_map=seg_node_map)
            # rendered_seg = rendered_seg[:, :, 0].copy()

            # confidence_u, confidence_v = self.compute_confidence(obs_depth_im, rendered_depth, rendered_seg,
            #                                                      inlier_thresh=0.03)

            tic = time.time()
            self.constrained_optimization(info)
            # print(f"optimization takes {time.time() - tic}s")
        return


    def create_scene_node(self, name, mesh, pose):
        m = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        return pyrender.Node(name=name, mesh=mesh, matrix=m @ pose)


    def render_nodes(self, seg_node_map: Dict[pyrender.Node, int], depth_only: bool = False):
        for node in seg_node_map:
            self.render_scene.add_node(node)

        if depth_only:
            rendered_depth = self.renderer.render(self.render_scene, flags=RenderFlags.DEPTH_ONLY)
            for node in seg_node_map:
                self.render_scene.remove_node(node)
            return rendered_depth

        rendered_seg, rendered_depth = self.renderer.render(self.render_scene, flags=RenderFlags.SEG,
                                                            seg_node_map=seg_node_map)
        rendered_seg = np.copy(rendered_seg[:, :, 0])
        for node in seg_node_map:
            self.render_scene.remove_node(node)
        return rendered_seg, rendered_depth


    def render_current_state(self):
        mapping = {'red': 1, 'green': 2, 'blue': 3}
        seg_node_map = dict()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            T = self.G.edges[u, v]['pose_list'][-1]
            mesh_node = self.create_scene_node(f"{color}-mash", self.rod_mesh, pose=T)
            seg_node_map[mesh_node] = mapping[color]
        return self.render_nodes(seg_node_map)


    def compute_confidence(self, obs_depth_im, rendered_depth_im, rendered_seg_im, inlier_thresh=0.01):
        seg_gpu = torch.from_numpy(rendered_seg_im).cuda()
        obs_depth_gpu = torch.from_numpy(obs_depth_im).cuda()
        rendered_depth_gpu = torch.from_numpy(rendered_depth_im).cuda()

        obs_pts, _ = self.back_projection(obs_depth_gpu)
        rendered_pts, (r_ys, r_xs) = self.back_projection(rendered_depth_gpu)
        pts_labels = seg_gpu[r_ys, r_xs]
        rendered_u = torch.sum(pts_labels == 1).cpu().item()
        rendered_v = torch.sum(pts_labels == 2).cpu().item()

        # nearest neighbor data association
        distances = torch.norm(rendered_pts.unsqueeze(1) - obs_pts.unsqueeze(0), dim=2)  # (N, M)
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
        obj_func, jac_func = self.objective_function_generator(sensor_measurement, sensor_status)

        init_values = np.zeros(3 * num_end_caps)
        for i in range(num_end_caps):
            init_values[(3 * i):(3 * i + 3)] = self.G.nodes[i]['pos_list'][-1]

        # be VERY CAREFUL when generating a lambda function in a for loop
        # https://stackoverflow.com/questions/45491376/
        # https://docs.python-guide.org/writing/gotchas/#late-binding-closures
        constraints = []
        for _, (u, v) in self.data_cfg['color_to_rod'].items():
            constraint = dict()
            constraint['type'] = 'eq'
            constraint['fun'], constraint['jac'] = self.endcap_constraint_generator(u, v, rod_length)
            constraints.append(constraint)

        rods = list(self.data_cfg['color_to_rod'].values())
        for i, (u, v) in enumerate(rods):
            p1 = self.G.nodes[u]['pos_list'][-1]
            p2 = self.G.nodes[v]['pos_list'][-1]
            for p, q in rods[i + 1:]:
                p3 = self.G.nodes[p]['pos_list'][-1]
                p4 = self.G.nodes[q]['pos_list'][-1]

                alpha1, alpha2 = self.compute_closest_pair_of_points(p1, p2, p3, p4)
                if alpha1 < 0 or alpha1 > 1 or alpha2 < 0 or alpha2 > 1:
                    continue

                constraint = dict()
                constraint['type'] = 'ineq'
                constraint['fun'], constraint['jac'] = self.rod_constraint_generator(u, v, p, q, alpha1, alpha2, rod_diameter=0.03)
                constraints.append(constraint)

        res = minimize(obj_func, init_values, jac=jac_func, method='SLSQP', constraints=constraints)
        assert res.success, "Optimization fail! Something must be wrong."

        for i in range(num_end_caps):
            # sanity check in case the constraint optimization fails
            if la.norm(res.x[(3 * i):(3 * i + 3)] - self.G.nodes[i]['pos_list'][-1]) < 0.1:
                self.G.nodes[i]['pos_list'][-1] = res.x[(3 * i):(3 * i + 3)].copy()
            else:
                print("Endcap position sanity check failed.")

        for u, v in self.data_cfg['color_to_rod'].values():
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
                factor = 0.2
                # if u_confidence > 0.5 and v_confidence > 0.5:
                #     factor = 0
                # if u_confidence < 0.2 or v_confidence <= 0.2:
                #     factor = 1

                u_pos = X[(3 * u):(3 * u + 3)]
                v_pos = X[(3 * v):(3 * v + 3)]
                estimated_length = la.norm(u_pos - v_pos)
                measured_length = sensor_measurement[sensor_id]['length'] / 1000
                binary_loss += factor * (estimated_length - measured_length)**2

            return unary_loss + binary_loss

        def jacobian_function(X):
            result = np.zeros_like(X)

            for i in range(len(self.data_cfg['node_to_color'])):
                x_hat, y_hat, z_hat = self.G.nodes[i]['pos_list'][-1]
                x, y, z = X[3*i : 3*i + 3]
                confidence = self.G.nodes[i].get('confidence', 1)
                result[3*i : 3*i + 3] = 2*confidence*np.array([x - x_hat, y - y_hat, z - z_hat])

            for sensor_id, (u, v) in self.data_cfg['sensor_to_tendon'].items():
                sensor_id = str(sensor_id)
                if not sensor_status[sensor_id] or sensor_measurement[sensor_id]['length'] <= 0:
                    continue

                u_confidence = self.G.nodes[u].get('confidence', 1)
                v_confidence = self.G.nodes[v].get('confidence', 1)
                factor = 0.2
                # if u_confidence > 0.5 and v_confidence > 0.5:
                #     factor = 0
                # if u_confidence < 0.2 or v_confidence <= 0.2:
                #     factor = 1

                u_x, u_y, u_z = X[(3 * u):(3 * u + 3)]
                v_x, v_y, v_z = X[(3 * v):(3 * v + 3)]
                l_est = np.sqrt((u_x - v_x)**2 + (u_y - v_y)**2 + (u_z - v_z)**2)
                l_gt = sensor_measurement[sensor_id]['length'] / 1000
                result[3*u : 3*u + 3] += 2*factor*(l_est - l_gt)*np.array([u_x - v_x, u_y - v_y, u_z - v_z]) / l_est
                result[3*v : 3*v + 3] += 2*factor*(l_est - l_gt)*np.array([v_x - u_x, v_y - u_y, v_z - u_z]) / l_est

            return result

        return objective_function, jacobian_function


    def endcap_constraint_generator(self, u, v, rod_length):

        def constraint_function(X):
            u_pos = X[3 * u: 3 * u + 3]
            v_pos = X[3 * v: 3 * v + 3]
            return la.norm(u_pos - v_pos) - rod_length

        def jacobian_function(X):
            u_x, u_y, u_z = X[3 * u : 3 * u + 3]
            v_x, v_y, v_z = X[3 * v : 3 * v + 3]
            C = np.sqrt((u_x - v_x)**2 + (u_y - v_y)**2 + (u_z - v_z)**2)  # denominator
            result = np.zeros_like(X)
            result[3 * u : 3 * u + 3] = np.array([u_x - v_x, u_y - v_y, u_z - v_z]) / C
            result[3 * v : 3 * v + 3] = np.array([v_x - u_x, v_y - u_y, v_z - u_z]) / C
            return result

        return constraint_function, jacobian_function


    def rod_constraint_generator(self, u, v, p, q, alpha1, alpha2, rod_diameter):
        try:
            p1 = self.G.nodes[u]['pos_list'][-2]
            p2 = self.G.nodes[v]['pos_list'][-2]
            p3 = self.G.nodes[p]['pos_list'][-2]
            p4 = self.G.nodes[q]['pos_list'][-2]
        except:
            p1 = self.G.nodes[u]['pos_list'][-1]
            p2 = self.G.nodes[v]['pos_list'][-1]
            p3 = self.G.nodes[p]['pos_list'][-1]
            p4 = self.G.nodes[q]['pos_list'][-1]
        p5 = alpha1 * p1 + (1 - alpha1) * p2  # closest point on (u, v)
        p6 = alpha2 * p3 + (1 - alpha2) * p4  # closest point on (p, q)
        v1 = p5 - p6
        v1 /= la.norm(v1)

        def constraint_function(X):
            q1 = X[3 * u : 3 * u + 3]
            q2 = X[3 * v : 3 * v + 3]
            q3 = X[3 * p : 3 * p + 3]
            q4 = X[3 * q : 3 * q + 3]
            q5 = alpha1 * q1 + (1 - alpha1) * q2  # closest point on (u, v)
            q6 = alpha2 * q3 + (1 - alpha2) * q4  # closest point on (p, q)
            v2 = q5 - q6
            return v1 @ v2 - rod_diameter

        def jacobian_function(X):
            result = np.zeros_like(X)
            result[3 * u : 3 * u + 3] = alpha1 * v1
            result[3 * v : 3 * v + 3] = (1 - alpha1) * v1
            result[3 * p : 3 * p + 3] = -alpha2 * v1
            result[3 * q : 3 * q + 3] = -(1 - alpha2) * v1
            return result

        return constraint_function, jacobian_function


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


    def projective_icp_cuda(self, obs_pts: torch.Tensor, rendered_pts: torch.Tensor,
                            init_pose: np.ndarray, max_distance=0.02, verbose=False):
        # ICP ref: https://www.youtube.com/watch?v=djnd502836w&t=781s
        # nearest neighbor data association
        distances = torch.norm(obs_pts.unsqueeze(1) - rendered_pts.unsqueeze(0), dim=2)  # (N, M)
        closest_distance, closest_indices = torch.min(distances, dim=1)  # find corresponded rendered pts for obs pts
        dist_mask = closest_distance < max_distance
        Q = obs_pts[dist_mask].T  # (3, K)
        P = rendered_pts[closest_indices][dist_mask].T  # (3, K)
        W = (1 - (closest_distance[dist_mask] / max_distance)**2)**2
        # W = 1 - closest_distance[dist_mask] / max_distance

        Q = Q.to(torch.float64)  # !!!
        P = P.to(torch.float64)  # !!!

        P_mean = P.mean(1, keepdims=True)  # (3, 1)
        Q_mean = Q.mean(1, keepdims=True)  # (3, 1)

        # https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx#L2367
        # Replace H = (Q - Q_mean) @ (P - P_mean).T with a potentially faster Einstein Summation
        H = torch.einsum('ji,jk->ik', W[:, None] * (Q - Q_mean).T, (P - P_mean).T)
        U, D, V = torch.svd(H.cpu())  # SVD on CPU is 100x faster than on GPU

        # ensure that R is in the right-hand coordinate system, very important!!!
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        d = torch.sign(torch.det(U @ V.T))
        R = U @ torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, d]
        ], dtype=torch.float64) @ V.T
        t = Q_mean - R.cuda() @ P_mean

        if verbose:
            error_before = torch.mean(torch.norm(Q - P, dim=0)).item()
            error_after = torch.mean(torch.norm(Q - (R @ P + t), dim=0)).item()
            print(f"{error_before = :.4f}, {error_after = :.4f}, delta = {error_before - error_after:.4f}")

        delta_T = np.eye(4, dtype=np.float64)
        delta_T[:3, :3] = R.cpu().numpy()
        delta_T[:3, 3:4] = t.cpu().numpy()
        return delta_T @ init_pose, Q.T, (R.cuda() @ P + t).T


    # https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines
    @staticmethod
    def compute_closest_pair_of_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray):
        """compute closest point pairs on two rods (p1, p2) and (p3, p4)

        Args:
            p1 (np.ndarray): start point of rod 1
            p2 (np.ndarray): end point of rod 1
            p3 (np.ndarray): start point of rod 2
            p4 (np.ndarray): end point of rod 2
        """
        v1 = p2 - p1
        l1 = la.norm(v1)
        v1 /= l1

        v2 = p4 - p3
        l2 = la.norm(v2)
        v2 /= l2

        v3 = np.cross(v2, v1)
        rhs = p3 - p1
        lhs = np.array([v1, -v2, v3]).T
        t1, t2, _ = la.solve(lhs, rhs)
        alpha1 = 1 - t1 / l1
        alpha2 = 1 - t2 / l2
        return alpha1, alpha2


    def visualize_closest_pair_of_points(self):

        def create_geometries_for_vis(p1, p2, color, radius=0.01):
            sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere1.paint_uniform_color(color)
            sphere1.translate(p1)

            sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere2.paint_uniform_color(color)
            sphere2.translate(p2)

            points = [p1.tolist(), p2.tolist()]
            lines = [[0, 1]]
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
            return [sphere1, sphere2, line_set]

        geometries = []
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            p1 = self.G.nodes[u]['pos_list'][-1]
            p2 = self.G.nodes[v]['pos_list'][-1]
            geometries.extend(create_geometries_for_vis(p1, p2, np.array(self.ColorDict[color]) / 255))

        rods = list(self.data_cfg['color_to_rod'].values())
        for i in range(len(rods)):
            u, v = rods[i]
            p1 = self.G.nodes[u]['pos_list'][-1]
            p2 = self.G.nodes[v]['pos_list'][-1]

            for j in range(i + 1, len(rods)):
                p, q = rods[j]
                p3 = self.G.nodes[p]['pos_list'][-1]
                p4 = self.G.nodes[q]['pos_list'][-1]

                alpha1, alpha2 = self.compute_closest_pair_of_points(p1, p2, p3, p4)
                p5 = alpha1 * p1 + (1 - alpha1) * p2
                p6 = alpha2 * p3 + (1 - alpha2) * p4
                geometries.extend(create_geometries_for_vis(p5, p6, [1, 1, 0], radius=0.005))

        o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--video_id", default="monday_roll15")
    # parser.add_argument("--video_id", default="socks6")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/struct_with_socks_new.ply")
    parser.add_argument("--top_end_cap_mesh_file", default="pcd/yale/end_cap_top_new.obj")
    parser.add_argument("--bottom_end_cap_mesh_file", default="pcd/yale/end_cap_bottom_new.obj")
    parser.add_argument("--first_frame_id", default=0, type=int)
    parser.add_argument("--max_iter", default=3, type=int)
    parser.add_argument("-v", "--visualize", default=False, action="store_true")
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video_id)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'color'))])

    # read data config
    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    # read rod point cloud
    assert os.path.isfile(args.rod_mesh_file), "rod geometry is not found!"
    rod_mesh_o3d = o3d.io.read_triangle_mesh(args.rod_mesh_file)
    rod_pcd = rod_mesh_o3d.sample_points_poisson_disk(1000)
    top_end_cap_mesh = o3d.io.read_triangle_mesh(args.top_end_cap_mesh_file)

    # read rod and end cap trimesh for for rendering
    rod_fuze_mesh = trimesh.load_mesh(args.rod_mesh_file)
    rod_mesh = pyrender.Mesh.from_trimesh(rod_fuze_mesh)
    top_end_cap_mesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(args.top_end_cap_mesh_file))
    bottom_end_cap_mesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(args.bottom_end_cap_mesh_file))

    # initialize tracker
    tracker = Tracker(data_cfg, rod_pcd, rod_mesh, [top_end_cap_mesh, bottom_end_cap_mesh])

    # initialize tracker with the first frame
    color_path = os.path.join(video_path, 'color', f'{prefixes[args.first_frame_id]}.png')
    depth_path = os.path.join(video_path, 'depth', f'{prefixes[args.first_frame_id]}.png')
    info_path = os.path.join(video_path, 'data', f'{prefixes[args.first_frame_id]}.json')
    # color_im = cv2.imread(color_path)
    color_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
    with open(info_path, 'r') as f:
        info = json.load(f)

    info['sensor_status'] = {key: True for key in info['sensors'].keys()}
    # info['sensor_status']['6'] = False
    # info['sensor_status']['7'] = False
    # info['sensor_status']['8'] = False

    tracker.initialize(color_im, depth_im, info, visualize=args.visualize, compute_hsv=False)
    data_cfg_module.write_config(tracker.data_cfg)

    # track frames
    os.makedirs(os.path.join(video_path, "scene_cloud"), exist_ok=True)
    os.makedirs(os.path.join(video_path, "estimation_cloud"), exist_ok=True)
    os.makedirs(os.path.join(video_path, "raw_estimation"), exist_ok=True)

    if args.visualize:
        cv2.namedWindow("estimation")
        cv2.moveWindow("estimation", 800, 100)
        rendered_seg, rendered_depth = tracker.render_current_state()
        vis_im = np.copy(color_im)
        for i, color in enumerate(['red', 'green', 'blue']):
            mask = rendered_depth.copy()
            mask[rendered_seg != i + 1] = 0
            vis_im[depth_im < mask] = Tracker.ColorDict[color]
        cv2.imshow("estimation", cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)

    data = dict()
    for idx in tqdm(range(args.first_frame_id + 1, len(prefixes))):
        prefix = prefixes[idx]
        color_path = os.path.join(video_path, 'color', f'{prefix}.png')
        depth_path = os.path.join(video_path, 'depth', f'{prefix}.png')
        info_path = os.path.join(video_path, 'data', f'{prefix}.json')
        data[prefix] = dict()

        # color_im = cv2.imread(color_path)
        color_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
        with open(info_path, 'r') as f:
            info = json.load(f)

        info['sensor_status'] = {key: True for key in info['sensors'].keys()}
        # info['sensor_status']['6'] = False
        # info['sensor_status']['7'] = False
        # info['sensor_status']['8'] = False

        data[prefix]['color_im'] = color_im
        data[prefix]['depth_im'] = depth_im
        data[prefix]['info'] = info

    for idx in tqdm(range(args.first_frame_id + 1, len(prefixes))):
        prefix = prefixes[idx]
        color_im = data[prefix]['color_im']
        depth_im = data[prefix]['depth_im']
        info = data[prefix]['info']

        tic = time.time()
        tracker.update(color_im, depth_im, info, args.max_iter)
        # print(f"update takes: {time.time() - tic}s")
        os.makedirs(os.path.join(video_path, 'estimation_rod_only'), exist_ok=True)

        if args.visualize:
            rendered_seg, rendered_depth = tracker.render_current_state()
            vis_im1 = np.copy(color_im)
            vis_im2 = np.zeros_like(color_im)
            for i, color in enumerate(['red', 'green', 'blue']):
                mask = rendered_depth.copy()
                mask[rendered_seg != i + 1] = 0
                vis_im1[depth_im < mask] = Tracker.ColorDict[color]
                vis_im2[mask > 0] = Tracker.ColorDict[color]
            vis_im = np.hstack([color_im, vis_im1, vis_im2])
            # vis_im = vis_im2
            vis_im_bgr = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
            cv2.imshow("estimation", vis_im_bgr)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit(0)
            cv2.imwrite(os.path.join(video_path, 'estimation_rod_only', f'{prefix}.png'), vis_im_bgr)

            # scene_pcd = perception_utils.create_pcd(depth_im, data_cfg['cam_intr'], color_im,
            #                                         depth_trunc=data_cfg['depth_trunc'])
            # robot_cloud = o3d.geometry.PointCloud()
            # for color, (u, v) in data_cfg['color_to_rod'].items():
            #     rod_pcd = copy.deepcopy(tracker.rod_pcd)
            #     rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
            #     rod_pose = copy.deepcopy(tracker.G.edges[u, v]['pose_list'][-1])
            #     rod_pcd.transform(rod_pose)
            #     robot_cloud += rod_pcd

            # estimation_cloud = robot_cloud
            # estimation_cloud = robot_cloud + scene_pcd
            # o3d.io.write_point_cloud(os.path.join(video_path, "scene_cloud", f"{idx:04d}.ply"), scene_pcd)
            # o3d.io.write_point_cloud(os.path.join(video_path, "estimation_cloud", f"{idx:04d}.ply"), estimation_cloud)

    # save rod poses and end cap positions to file
    pose_output_folder = os.path.join(video_path, "poses")
    os.makedirs(pose_output_folder, exist_ok=True)
    for color, (u, v) in tracker.data_cfg['color_to_rod'].items():
        np.save(os.path.join(pose_output_folder, f'{color}.npy'), np.array(tracker.G.edges[u, v]['pose_list']))
        np.save(os.path.join(pose_output_folder, f'{u}_pos.npy'), np.array(tracker.G.nodes[u]['pos_list']))
        np.save(os.path.join(pose_output_folder, f'{v}_pos.npy'), np.array(tracker.G.nodes[v]['pos_list']))
