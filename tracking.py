import os
import time
import copy
import importlib
import pprint
import json
from argparse import ArgumentParser
from typing import List, Dict

import numpy as np
import numpy.linalg as la
import cv2
import networkx as nx
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from tqdm import tqdm
import trimesh
from trimesh.sample import sample_surface_even
import pyrender
from pyrender.constants import RenderFlags

from perception_utils import create_pcd, plane_detection_ransac, vis_pcd


class Tracker:

    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }

    def __init__(self, cfg, data_cfg, rod_pcd=None, rod_mesh=None, endcap_meshes=None):
        self.cfg = cfg
        self.data_cfg = data_cfg
        pprint.pprint(self.data_cfg)
        self.initialized = False

        self.endcap_meshes = endcap_meshes
        self.rod_pcd = copy.deepcopy(rod_pcd)
        self.rod_mesh = rod_mesh
        self.rod_length = self.data_cfg['rod_length']

        H, W = self.data_cfg['im_h'], self.data_cfg['im_w']
        self.obs_vis = np.empty((self.data_cfg['im_h'], self.data_cfg['im_w'], 3), dtype=np.uint8)

        # set up pyrender scene
        self.renderer = pyrender.OffscreenRenderer(W // self.cfg.render_scale, H // self.cfg.render_scale)
        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx = cam_intr[0, 0] / self.cfg.render_scale
        fy = cam_intr[1, 1] / self.cfg.render_scale
        cx = cam_intr[0, 2] / self.cfg.render_scale
        cy = cam_intr[1, 2] / self.cfg.render_scale
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
    

    def initialize(self, rgb_im: np.ndarray, depth_im: np.ndarray, info: dict):
        self.rgb_im = rgb_im
        self.depth_im = depth_im
        self.hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)

        cam_intr = np.asarray(self.data_cfg['cam_intr'])
        fx, fy, cx, cy = cam_intr[0, 0], cam_intr[1, 1], cam_intr[0, 2], cam_intr[1, 2]

        scene_pcd = create_pcd(depth_im, self.data_cfg['cam_intr'], rgb_im, depth_trunc=self.data_cfg['depth_trunc'])
        if 'cam_extr' not in self.data_cfg:
            plane_frame, _ = plane_detection_ransac(scene_pcd, inlier_thresh=0.003, visualize=self.cfg.visualize)
            self.data_cfg['cam_extr'] = np.round(plane_frame, decimals=3)

        if 'init_endcap_pos' not in self.data_cfg:
            self.data_cfg['init_endcap_pos'] = dict()
            rgb_im_copy = np.copy(rgb_im)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(rgb_im_copy)

            global endcap_id
            endcap_id = 0
            endcap_color = self.G.nodes[endcap_id]['color']
            plt.get_current_fig_manager().set_window_title(f"Select #{endcap_id} endcap ({endcap_color})")

            def get_point(event):
                if event.xdata and event.ydata:
                    x = int(round(event.xdata))
                    y = int(round(event.ydata))
                    rgb = list(rgb_im_copy[y, x])
                    hsv = list(self.hsv_im[y, x])
                    Z = float(self.depth_im[y, x])
                    X = float((x - cx) * Z / fx)
                    Y = float((y - cy) * Z / fy)
                    if Z == 0:
                        print(f"2D pos: ({x}, {y}), 3D pos: N/A, rgb: {rgb}, hsv: {hsv}")
                    else:
                        print(f"2D pos: ({x}, {y}), 3D pos: ({X:.3f}, {Y:.3f}, {Z:.3f}), rgb: {rgb}, hsv: {hsv}")
                    return {'pos_2d': [x, y], 'pos_3d': [X, Y, Z], 'rgb': rgb, 'hsv': hsv}
            
            def mouse_click_callback(event):
                global endcap_id
                endcap_color = self.G.nodes[endcap_id]['color']
                point = get_point(event)
                if point and point['pos_3d'][-1] != 0:
                    print("===========================================================================================")
                    print(f"#{endcap_id} endcap ({self.G.nodes[endcap_id]['color']})")
                    pprint.pprint(point)
                    print("===========================================================================================")
                    self.data_cfg['init_endcap_pos'][str(endcap_id)] = point['pos_3d']

                    # check whether this point is in the HSV range of the endcap color
                    lowerb, upperb = self.data_cfg['hsv_ranges'][endcap_color]
                    hsv = point['hsv']
                    if np.any(hsv < np.asarray(lowerb)) or np.any(hsv > np.asarray(upperb)):
                        info['node_confidence'][endcap_id] = 0
                    else:
                        info['node_confidence'][endcap_id] = 1

                    endcap_id += 1
                    if endcap_id < len(self.G.nodes):
                        endcap_color = self.G.nodes[endcap_id]['color']
                        plt.get_current_fig_manager().set_window_title(f"Select #{endcap_id} endcap ({endcap_color})")
                    else:
                        plt.close()

            fig.canvas.mpl_connect('motion_notify_event', get_point)
            fig.canvas.mpl_connect('button_press_event', mouse_click_callback)
            plt.show()
            
        # -------------------- optimize once -------------------- 
        for _, (u, v) in self.data_cfg['color_to_rod'].items():
            self.G.nodes[u]['pos_list'].append(self.data_cfg['init_endcap_pos'][str(u)])
            self.G.nodes[v]['pos_list'].append(self.data_cfg['init_endcap_pos'][str(v)])

        add_physical_constraints = self.cfg.add_physical_constraints
        self.cfg.add_physical_constraints = False
        self.constrained_optimization(info)
        self.cfg.add_physical_constraints = add_physical_constraints
        # -------------------------------------------------------

        for _, (u, v) in self.data_cfg['color_to_rod'].items():
            self.data_cfg['init_endcap_pos'][str(u)] = self.G.nodes[u]['pos_list'][0].tolist()
            self.data_cfg['init_endcap_pos'][str(v)] = self.G.nodes[v]['pos_list'][0].tolist()

        if self.cfg.visualize:
            pcd = self.get_3d_vis()
            o3d.visualization.draw_geometries([pcd])

        self.initialized = True


    def compute_obs_pts(self, depth_im, mask):
        fx, fy = self.data_cfg['cam_intr'][0][0], self.data_cfg['cam_intr'][1][1]
        cx, cy = self.data_cfg['cam_intr'][0][2], self.data_cfg['cam_intr'][1][2]

        ys, xs = np.nonzero((depth_im > 0) & mask)
        pts = np.empty([len(ys), 3], dtype=np.float32)
        pts[:, 2] = depth_im[ys, xs]
        pts[:, 0] = (xs - cx) / fx * pts[:, 2]
        pts[:, 1] = (ys - cy) / fy * pts[:, 2]
        return pts


    def filter_obs_pts(self, obs_pts, color, radius=0.1, thresh=0.01):
        u, v = self.data_cfg['color_to_rod'][color]
        u_pos = self.G.nodes[u]['pos_list'][-1]
        u_dist_sqr = np.sum((obs_pts - u_pos)**2, axis=1)
        v_pos = self.G.nodes[v]['pos_list'][-1]
        v_dist_sqr = np.sum((obs_pts - v_pos)**2, axis=1)

        u_mask = u_dist_sqr < v_dist_sqr
        u_pts = obs_pts[u_mask & (u_dist_sqr < radius**2)]
        v_pts = obs_pts[(~u_mask) & (v_dist_sqr < radius**2)]

        # filter points based on depth
        if u_pts.shape[0] > 0:
            zs = u_pts[:, 2]
            z_min = np.percentile(zs, q=10)
            u_pts = u_pts[zs < z_min + thresh]

        if v_pts.shape[0] > 0:
            zs = v_pts[:, 2]
            z_min = np.percentile(zs, q=10)
            v_pts = v_pts[zs < z_min + thresh]

        return u_pts, v_pts


    def update(self, rgb_im, depth_im, info):
        assert self.initialized, "[Error] You must first initialize the tracker!"
        self.rgb_im = rgb_im
        self.depth_im = depth_im

        # get observed 3D points for both end caps of each rod
        obs_pts_dict = dict()
        model_pcd_dict = dict()
        mesh_nodes = dict()

        self.obs_vis = np.zeros_like(self.obs_vis)
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            color_im_hsv = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)
            hsv_mask = cv2.inRange(
                color_im_hsv,
                lowerb=tuple(self.data_cfg['hsv_ranges'][color][0]),
                upperb=tuple(self.data_cfg['hsv_ranges'][color][1])
            ).astype(np.bool8)
            # if color == 'red':
            #     hsv_mask |= cv2.inRange(color_im_hsv, lowerb=(0, 100, 25), upperb=(3, 255, 100)).astype(np.bool8)
            obs_pts = self.compute_obs_pts(depth_im, hsv_mask)
            obs_pts_dict[color] = obs_pts

            prev_pose = np.copy(self.G.edges[u, v]['pose_list'][-1])
            for i, node in enumerate([u, v]):
                model_pcd = copy.deepcopy(self.endcap_meshes[i])
                model_pcd.transform(prev_pose)
                _, pt_map = model_pcd.hidden_point_removal([0, 0, 0], radius=0.001)
                model_pcd = model_pcd.select_by_index(pt_map, invert=True)
                model_pcd_dict[node] = model_pcd

            self.G.edges[u, v]['pose_list'].append(prev_pose)
            self.G.nodes[u]['pos_list'].append(prev_pose[:3, 3] + prev_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[v]['pos_list'].append(prev_pose[:3, 3] - prev_pose[:3, 2] * self.rod_length / 2)

            # prev_pose = self.G.edges[u, v]['pose_list'][-1].copy()
            # mesh_nodes[self.create_scene_node(f"node-{u}", self.endcap_meshes[0], prev_pose)] = u + 1
            # mesh_nodes[self.create_scene_node(f"node-{v}", self.endcap_meshes[1], prev_pose)] = v + 1

            if color == 'red':
                self.obs_vis[hsv_mask, 0] = 255
            elif color == 'green':
                self.obs_vis[hsv_mask, 1] = 255
            elif color == 'blue':
                self.obs_vis[hsv_mask, 2] = 255

        # rendered_seg, rendered_depth = self.render_nodes(mesh_nodes, depth_only=False)
        # rendered_seg_gpu = torch.from_numpy(rendered_seg).cuda()
        # rendered_depth_gpu = torch.from_numpy(rendered_depth).cuda().to(torch.float32)
        # for color, (u, v) in self.data_cfg['color_to_rod'].items():
        #     u_pcd = o3d.geometry.PointCloud()
        #     u_pcd.points = o3d.utility.Vector3dVector(self.compute_obs_pts(rendered_depth_gpu, (rendered_seg_gpu == u + 1)))
        #     v_pcd = o3d.geometry.PointCloud()
        #     v_pcd.points = o3d.utility.Vector3dVector(self.compute_obs_pts(rendered_depth_gpu, (rendered_seg_gpu == v + 1)))
        #     model_pcd_dict[u] = u_pcd
        #     model_pcd_dict[v] = v_pcd

        for iter, max_distance in enumerate(self.cfg.max_correspondence_distances):
            prev_poses = {}
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                prev_poses[color] = np.copy(self.G.edges[u, v]['pose_list'][-1])

            # ================================== prediction step ==================================
            tic = time.time()
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                # u_rendered_pts = self.compute_obs_pts(rendered_depth_gpu, (rendered_seg_gpu == u + 1))
                # v_rendered_pts = self.compute_obs_pts(rendered_depth_gpu, (rendered_seg_gpu == v + 1))
                u_rendered_pts = np.asarray(model_pcd_dict[u].points)
                v_rendered_pts = np.asarray(model_pcd_dict[v].points)

                obs_pts = obs_pts_dict[color]
                # filter observed points based on z coordinates
                if self.cfg.filter_observed_pts:
                    u_obs_pts, v_obs_pts = self.filter_obs_pts(obs_pts, color, radius=max_distance, thresh=0.02)
                else:
                    u_obs_pts, v_obs_pts = self.filter_obs_pts(obs_pts, color, radius=max_distance, thresh=np.inf)

                # add dummy points at the previous endcap position
                if self.cfg.add_dummy_points:
                    u_dummy = self.G.nodes[u]['pos_list'][-1]
                    v_dummy = self.G.nodes[v]['pos_list'][-1]

                    obs_pts = np.vstack([u_obs_pts, v_obs_pts, u_dummy, v_dummy])
                    obs_w = np.ones(obs_pts.shape[0])
                    obs_w[-2:] = self.cfg.dummy_weights * self.cfg.num_dummy_points

                    rendered_pts = np.vstack([u_rendered_pts, v_rendered_pts, u_dummy, v_dummy])
                    rendered_w = np.ones(rendered_pts.shape[0])
                    rendered_w[-2:] = self.cfg.dummy_weights * self.cfg.num_dummy_points

                delta_T = self.register(obs_pts, rendered_pts, obs_w, rendered_w, max_distance)
                rod_pose = delta_T @ prev_poses[color]

                self.G.edges[u, v]['pose_list'][-1] = rod_pose
                self.G.nodes[u]['pos_list'][-1] = rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2
                self.G.nodes[v]['pos_list'][-1] = rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2

                # self.G.nodes[u]['confidence'] = self.compute_confidence(u_obs_pts, u_rendered_pts, num_thresh=50, inlier_thresh=0.02)
                # self.G.nodes[v]['confidence'] = self.compute_confidence(v_obs_pts, v_rendered_pts, num_thresh=50, inlier_thresh=0.02)

            # print(f"point registration takes {time.time() - tic}s")

            # ================================== correction step ==================================
            if self.cfg.optimize_every_n_iters != 0 and iter % self.cfg.optimize_every_n_iters == 0:
                tic = time.time()
                self.constrained_optimization(info)
                # print(f"optimization takes {time.time() - tic}s")
            
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                prev_pose = prev_poses[color]
                curr_pose = self.G.edges[u, v]['pose_list'][-1]
                delta_T = curr_pose @ la.inv(prev_pose)
                model_pcd_dict[u].transform(delta_T)
                model_pcd_dict[v].transform(delta_T)
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

        scale = self.cfg.render_scale
        if depth_only:
            rendered_depth = self.renderer.render(self.render_scene, flags=RenderFlags.DEPTH_ONLY)
            for node in seg_node_map:
                self.render_scene.remove_node(node)
            if scale != 1:
                H, W = rendered_depth.shape
                rendered_depth = cv2.resize(rendered_depth, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST)
            return rendered_depth

        rendered_seg, rendered_depth = self.renderer.render(self.render_scene, flags=RenderFlags.SEG,
                                                            seg_node_map=seg_node_map)
        rendered_seg = np.copy(rendered_seg[:, :, 0])
        for node in seg_node_map:
            self.render_scene.remove_node(node)
        if scale != 1:
            H, W = rendered_depth.shape
            rendered_depth = cv2.resize(rendered_depth, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST)
            rendered_seg = cv2.resize(rendered_seg, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST)
        return rendered_seg, rendered_depth


    def render_current_state(self):
        mapping = {'red': 1, 'green': 2, 'blue': 3}
        seg_node_map = dict()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            pose = self.G.edges[u, v]['pose_list'][-1]
            mesh_node = self.create_scene_node(f"{color}-mash", self.rod_mesh, pose=pose)
            seg_node_map[mesh_node] = mapping[color]
        return self.render_nodes(seg_node_map)


    # def compute_confidence(self, obs_pts, rendered_pts, num_thresh=50, inlier_thresh=0.01):
    #     N, M  = obs_pts.shape[0], rendered_pts.shape[0]
    #     if N < num_thresh or M < num_thresh:
    #         return 0

    #     distances = torch.norm(rendered_pts.unsqueeze(1) - obs_pts.unsqueeze(0), dim=2)  # (N, M)
    #     tp = 0

    #     closest_distance, _ = torch.min(distances, dim=1)
    #     tp += torch.sum(closest_distance < inlier_thresh)

    #     closest_distance, _ = torch.min(distances, dim=0)
    #     tp += torch.sum(closest_distance < inlier_thresh)

    #     f1 = tp / (N + M)
    #     return f1.item()


    def constrained_optimization(self, info):
        rod_length = self.data_cfg['rod_length']
        num_endcaps = 2 * self.data_cfg['num_rods']  # a rod has two end caps

        obj_func, jac_func = self.objective_function_generator(info)

        init_values = np.zeros(3 * num_endcaps)
        for i in range(num_endcaps):
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

        if self.cfg.add_ground_constraints:
            for u, color in enumerate(self.data_cfg['node_to_color']):
                # if color == 'blue':
                #     continue
                constraint = dict()
                constraint['type'] = 'ineq'
                constraint['fun'], constraint['jac'] = self.ground_constraint_generator(u)
                constraints.append(constraint)

        if self.cfg.add_physical_constraints:
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
                    constraint['fun'], constraint['jac'] = self.rod_constraint_generator(u, v, p, q, alpha1, alpha2)
                    constraints.append(constraint)

        res = minimize(obj_func, init_values, jac=jac_func, method='SLSQP', constraints=constraints)
        if not res.success:
            print("Joint optimization fail! Something must be wrong.")

        for i in range(num_endcaps):
            # sanity check in case the constraint optimization fails
            if la.norm(res.x[(3 * i):(3 * i + 3)] - self.G.nodes[i]['pos_list'][-1]) > 0.1:
                print("Endcap position sanity check failed.")
            self.G.nodes[i]['pos_list'][-1] = res.x[(3 * i):(3 * i + 3)].copy()

        for u, v in self.data_cfg['color_to_rod'].values():
            if self.G.edges[u, v]['pose_list']:
                prev_rod_pose = self.G.edges[u, v]['pose_list'][-1]
            else:
                prev_rod_pose = None
            u_pos = self.G.nodes[u]['pos_list'][-1]
            v_pos = self.G.nodes[v]['pos_list'][-1]
            curr_endcap_centers = np.vstack([u_pos, v_pos])
            optimized_pose = self.estimate_rod_pose_from_endcap_centers(curr_endcap_centers, prev_rod_pose)
            if self.G.edges[u, v]['pose_list']:
                self.G.edges[u, v]['pose_list'][-1] = optimized_pose
            else:
                self.G.edges[u, v]['pose_list'].append(optimized_pose)


    def ground_constraint_generator(self, u):
        cam_to_world = la.inv(self.data_cfg['cam_extr'])
        R = cam_to_world[:3, :3]
        t = cam_to_world[:3, 3]

        def constraint_function(X):
            u_pos = X[3 * u: 3 * u + 3]
            u_world_pos = R @ u_pos + t
            u_z = u_world_pos[2]
            return u_z - self.data_cfg['rod_diameter'] / 2 + 0.01

        def jacobian_function(X):
            result = np.zeros_like(X)
            result[3 * u : 3 * u + 3] = R[2]
            return result

        return constraint_function, jacobian_function


    def objective_function_generator(self, info):
        sensor_measurement = info['sensors']
        node_confidence = info['node_confidence']

        def objective_function(X):
            unary_loss = 0
            for i in range(len(self.data_cfg['node_to_color'])):
                pos_est = X[(3 * i):(3 * i + 3)]
                pos_old = self.G.nodes[i]['pos_list'][-1]
                confidence = node_confidence.get(i, 1)
                # confidence = self.G.nodes[i].get('confidence', 1)
                # confidence = 1
                unary_loss += confidence * np.sum((pos_est - pos_old)**2)

            binary_loss = 0
            for sensor_id, (u, v) in self.data_cfg['sensor_to_tendon'].items():
                sensor_id = str(sensor_id)
                if not self.data_cfg["sensor_status"][sensor_id] or sensor_measurement[sensor_id]['length'] <= 0:
                    continue

                factor = 0.2
                # c_u = self.G.nodes[u].get('confidence', 1)
                # c_v = self.G.nodes[v].get('confidence', 1)
                # if c_u > 0.5 and c_v > 0.5:
                #     factor = 0.05
                # elif c_u > 0.2 and c_v > 0.2:
                #     factor *= (1 - 0.5*(c_u + c_v))

                u_pos = X[(3 * u):(3 * u + 3)]
                v_pos = X[(3 * v):(3 * v + 3)]
                estimated_length = la.norm(u_pos - v_pos)
                measured_length = sensor_measurement[sensor_id]['length'] / self.data_cfg['cable_length_scale']

                # sanity check
                if np.abs(estimated_length - measured_length) > 0.1:
                    factor = 0

                binary_loss += factor * (estimated_length - measured_length)**2

            return unary_loss + binary_loss

        def jacobian_function(X):
            result = np.zeros_like(X)

            for i in range(len(self.data_cfg['node_to_color'])):
                pos_est = X[3 * i : 3 * i + 3]
                pos_old = self.G.nodes[i]['pos_list'][-1]
                confidence = node_confidence.get(i, 1)
                # confidence = self.G.nodes[i].get('confidence', 1)
                # confidence = 1
                result[3 * i : 3 * i + 3] = 2 * confidence * (pos_est - pos_old)

            for sensor_id, (u, v) in self.data_cfg['sensor_to_tendon'].items():
                sensor_id = str(sensor_id)
                if not self.data_cfg['sensor_status'][sensor_id] or sensor_measurement[sensor_id]['length'] <= 0:
                    continue

                c_u = self.G.nodes[u].get('confidence', 1)
                c_v = self.G.nodes[v].get('confidence', 1)
                factor = 0.2
                # if c_u > 0.5 and c_v > 0.5:
                #     factor = 0.05
                # elif c_u > 0.2 and c_v > 0.2:
                #     factor *= (1 - 0.5*(c_u + c_v))

                u_pos = X[(3 * u):(3 * u + 3)]
                v_pos = X[(3 * v):(3 * v + 3)]
                l_est = la.norm(u_pos - v_pos)
                l_gt = sensor_measurement[sensor_id]['length'] / self.data_cfg['cable_length_scale']

                # sanity check
                if np.abs(l_est - l_gt) > 0.1:
                    factor = 0

                result[3 * u : 3 * u + 3] += 2 * factor * (l_est - l_gt) * (u_pos - v_pos) / (l_est + 1e-12)
                result[3 * v : 3 * v + 3] += 2 * factor * (l_est - l_gt) * (v_pos - u_pos) / (l_est + 1e-12)

            return result

        return objective_function, jacobian_function


    def endcap_constraint_generator(self, u, v, rod_length):

        def constraint_function(X):
            u_pos = X[3 * u: 3 * u + 3]
            v_pos = X[3 * v: 3 * v + 3]
            return la.norm(u_pos - v_pos) - rod_length

        def jacobian_function(X):
            u_pos = X[3 * u : 3 * u + 3]
            v_pos = X[3 * v : 3 * v + 3]
            C = la.norm(u_pos - v_pos) + 1e-12
            result = np.zeros_like(X)
            result[3 * u : 3 * u + 3] = (u_pos - v_pos) / C
            result[3 * v : 3 * v + 3] = (v_pos - u_pos) / C
            return result

        return constraint_function, jacobian_function


    def rod_constraint_generator(self, u, v, p, q, alpha1, alpha2):
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
        v1 /= (la.norm(v1) + 1e-12)

        def constraint_function(X):
            q1 = X[3 * u : 3 * u + 3]
            q2 = X[3 * v : 3 * v + 3]
            q3 = X[3 * p : 3 * p + 3]
            q4 = X[3 * q : 3 * q + 3]
            q5 = alpha1 * q1 + (1 - alpha1) * q2  # closest point on (u, v)
            q6 = alpha2 * q3 + (1 - alpha2) * q4  # closest point on (p, q)
            v2 = q5 - q6
            return v1 @ v2 - self.data_cfg['rod_diameter']

        def jacobian_function(X):
            result = np.zeros_like(X)
            result[3 * u : 3 * u + 3] = alpha1 * v1
            result[3 * v : 3 * v + 3] = (1 - alpha1) * v1
            result[3 * p : 3 * p + 3] = -alpha2 * v1
            result[3 * q : 3 * q + 3] = -(1 - alpha2) * v1
            return result

        return constraint_function, jacobian_function


    @staticmethod
    def estimate_rod_pose_from_endcap_centers(curr_endcap_centers, prev_rod_pose=None):
        curr_rod_pos = (curr_endcap_centers[0] + curr_endcap_centers[1]) / 2
        curr_z_dir = curr_endcap_centers[0] - curr_endcap_centers[1]
        curr_z_dir /= la.norm(curr_z_dir)

        if prev_rod_pose is None:
            prev_rod_pose = np.eye(4)

        prev_rot = prev_rod_pose[:3, :3]
        prev_z_dir = prev_rot[:, 2]

        delta_rot = np.eye(3)
        cos_dist = prev_z_dir @ curr_z_dir
        if not np.allclose(cos_dist, 1):
            axis = np.cross(prev_z_dir, curr_z_dir)
            axis = axis / la.norm(axis)
            angle = np.arccos(cos_dist)
            delta_rot = Rotation.from_rotvec(angle * axis).as_matrix()

        curr_rod_pose = np.eye(4)
        curr_rod_pose[:3, :3] = delta_rot @ prev_rot
        curr_rod_pose[:3, 3] = curr_rod_pos
        return curr_rod_pose


    def register(self, obs_pts, rendered_pts, obs_w=None, rendered_w=None, max_distance=0.02):
        if obs_w is None:
            obs_w = np.ones(obs_pts.shape[0])
        if rendered_w is None:
            rendered_w = np.ones(rendered_pts.shape[0])

        # ICP ref: https://www.youtube.com/watch?v=djnd502836w&t=781s
        r_tree = KDTree(rendered_pts)
        closest_distance, closest_indices = r_tree.query(obs_pts, k=1, distance_upper_bound=max_distance)
        mask = (closest_distance != np.inf)
        Q1 = obs_pts[mask]
        P1 = rendered_pts[closest_indices[mask]]
        W1 = (1 - (closest_distance[mask] / max_distance)**2)**2 * obs_w[mask] * rendered_w[closest_indices[mask]]

        o_tree = KDTree(obs_pts)
        closest_distance, closest_indices = o_tree.query(rendered_pts, k=1, distance_upper_bound=max_distance)
        mask = (closest_distance != np.inf)
        Q2 = obs_pts[closest_indices[mask]]
        P2 = rendered_pts[mask]
        W2 = (1 - (closest_distance[mask] / max_distance)**2)**2 * obs_w[closest_indices[mask]] * rendered_w[mask]

        Q = np.vstack([Q1, Q2])
        P = np.vstack([P1, P2])
        W = np.hstack([W1, W2])

        P_mean = P.mean(0)
        Q_mean = Q.mean(0)

        # https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx#L2367
        H = (W[:, None] * (Q - Q_mean)).T @ (P - P_mean)
        U, _, V_t = la.svd(H)

        # ensure that R is in the right-hand coordinate system, very important!!!
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        d = np.sign(la.det(U @ V_t))
        U[:, -1] = d * U[:, -1]
        R = U @ V_t
        t = Q_mean - R @ P_mean
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


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


    def get_2d_vis(self):
        rendered_seg, rendered_depth = tracker.render_current_state()
        H, W = self.data_cfg['im_h'], self.data_cfg['im_w']
        vis_im1 = np.copy(self.rgb_im)
        vis_im2 = np.zeros_like(self.rgb_im)
        for i, color in enumerate(['red', 'green', 'blue']):
            mask = rendered_depth.copy()
            mask[rendered_seg != i + 1] = 0
            vis_im1[depth_im < mask] = Tracker.ColorDict[color]
            vis_im2[mask > 0] = Tracker.ColorDict[color]
        vis_im = np.empty((H*2, W*2, 3), dtype=np.uint8)
        vis_im[:H, :W] = self.rgb_im
        vis_im[H:, :W] = tracker.obs_vis
        vis_im[:H, W:] = vis_im1
        vis_im[H:, W:] = vis_im2
        return vis_im


    def get_3d_vis(self):
        robot_pcd = o3d.geometry.PointCloud()
        for color, (u, v) in data_cfg['color_to_rod'].items():
            rod_pcd = copy.deepcopy(tracker.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
            rod_pose = np.copy(tracker.G.edges[u, v]['pose_list'][-1])
            rod_pcd.transform(rod_pose)
            robot_pcd += rod_pcd

        scene_pcd = create_pcd(self.depth_im, data_cfg['cam_intr'], self.rgb_im, depth_trunc=data_cfg['depth_trunc'])
        robot_pts = np.asarray(robot_pcd.points)
        bbox = o3d.geometry.AxisAlignedBoundingBox()
        bbox.min_bound = robot_pts.min(axis=0) - 0.05
        bbox.max_bound = robot_pts.max(axis=0) + 0.05
        scene_pcd = scene_pcd.crop(bbox)
        return robot_pcd + scene_pcd


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--video", default="monday_roll15")
    parser.add_argument("--method", default="proposed")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/struct_with_socks_new.ply")
    parser.add_argument("--top_endcap_mesh_file", default="pcd/yale/end_cap_top_new.obj")
    parser.add_argument("--bottom_endcap_mesh_file", default="pcd/yale/end_cap_bottom_new.obj")
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=10000, type=int)
    parser.add_argument("--render_scale", default=2, type=int)
    parser.add_argument("--max_correspondence_distances", default=[0.3, 0.15, 0.1, 0.06, 0.03], type=float, nargs="+")
    parser.add_argument("--add_dummy_points", action="store_true")
    parser.add_argument("--num_dummy_points", type=int, default=50)
    parser.add_argument("--dummy_weights", type=float, default=0.1)
    parser.add_argument("--optimize_every_n_iters", type=int, default=1)
    parser.add_argument("--filter_observed_pts", action="store_true")
    parser.add_argument("--add_physical_constraints", action="store_true")
    parser.add_argument("--add_ground_constraints", action="store_true")
    parser.add_argument("--save", action="store_true", help="save observation and qualitative results")
    parser.add_argument("-v", "--visualize", action="store_true")
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'color'))])

    # read data config
    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    # read rod point cloud
    assert os.path.isfile(args.rod_mesh_file), "rod geometry is not found!"
    rod_mesh_o3d = o3d.io.read_triangle_mesh(args.rod_mesh_file)
    rod_pcd = rod_mesh_o3d.sample_points_poisson_disk(1000)

    # read rod and end cap trimesh for for rendering
    rod_mesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(args.rod_mesh_file))
    # top_endcap_mesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(args.top_endcap_mesh_file))
    # bottom_endcap_mesh = pyrender.Mesh.from_trimesh(trimesh.load_mesh(args.bottom_endcap_mesh_file))

    top_endcap_mesh = trimesh.load_mesh(args.top_endcap_mesh_file)
    top_endcap_pts, _ = sample_surface_even(top_endcap_mesh, 300)
    top_endcap_pcd = o3d.geometry.PointCloud()
    top_endcap_pcd.points = o3d.utility.Vector3dVector(top_endcap_pts)
    bottom_endcap_mesh = trimesh.load_mesh(args.bottom_endcap_mesh_file)
    bottom_endcap_pts, _ = sample_surface_even(bottom_endcap_mesh, 300)
    bottom_endcap_pcd = o3d.geometry.PointCloud()
    bottom_endcap_pcd.points = o3d.utility.Vector3dVector(bottom_endcap_pts)
    # o3d.visualization.draw_geometries([top_endcap_pcd, bottom_endcap_pcd])

    # initialize tracker
    tracker = Tracker(args, data_cfg, rod_pcd, rod_mesh, [top_endcap_pcd, bottom_endcap_pcd])
    # tracker = Tracker(args, data_cfg, rod_pcd, rod_mesh, [top_endcap_mesh, bottom_endcap_mesh])

    # initialize tracker with the first frame
    color_path = os.path.join(video_path, 'color', f'{prefixes[args.start_frame]}.png')
    depth_path = os.path.join(video_path, 'depth', f'{prefixes[args.start_frame]}.png')
    info_path = os.path.join(video_path, 'data', f'{prefixes[args.start_frame]}.json')
    rgb_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
    with open(info_path, 'r') as f:
        info = json.load(f)
    info['node_confidence'] = dict()

    tracker.initialize(rgb_im, depth_im, info)
    data_cfg_module.write_config(tracker.data_cfg)

    if args.save:
        os.makedirs(os.path.join(video_path, f'estimation_vis-{args.method}'), exist_ok=True)
        os.makedirs(os.path.join(video_path, f"estimation_cloud-{args.method}"), exist_ok=True)

    end_frame = min(len(prefixes), args.end_frame)
    total_time = 0
    for idx in tqdm(range(args.start_frame, end_frame)):
        prefix = prefixes[idx]
        color_path = os.path.join(video_path, 'color', f'{prefix}.png')
        depth_path = os.path.join(video_path, 'depth', f'{prefix}.png')
        info_path = os.path.join(video_path, 'data', f'{prefix}.json')

        rgb_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
        with open(info_path, 'r') as f:
            info = json.load(f)
        info['node_confidence'] = dict()

        tic = time.time()
        tracker.update(rgb_im, depth_im, info)
        total_time += time.time() - tic

        if args.visualize:
            vis_im = tracker.get_2d_vis()
            vis_im_bgr = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)
            cv2.imshow("estimation", vis_im_bgr)
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit(0)

            if args.save:
                cv2.imwrite(os.path.join(video_path, f'estimation_vis-{args.method}', f'{prefix}.jpg'), vis_im_bgr)
                o3d.io.write_point_cloud(os.path.join(video_path, f"estimation_cloud-{args.method}", f"{idx:04d}.pcd"), tracker.get_3d_vis())
    print(f"FPS: {(end_frame - args.start_frame + 1e-12) / total_time}")

    # save rod poses and end cap positions to file
    if args.save:
        pose_output_folder = os.path.join(video_path, f"poses-{args.method}")
        os.makedirs(pose_output_folder, exist_ok=True)
        for color, (u, v) in tracker.data_cfg['color_to_rod'].items():
            np.save(os.path.join(pose_output_folder, f'{color}.npy'), np.array(tracker.G.edges[u, v]['pose_list'])[1:])
            np.save(os.path.join(pose_output_folder, f'{u}_pos.npy'), np.array(tracker.G.nodes[u]['pos_list'])[1:])
            np.save(os.path.join(pose_output_folder, f'{v}_pos.npy'), np.array(tracker.G.nodes[v]['pos_list'])[1:])
