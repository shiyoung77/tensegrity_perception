import os
import time
import copy
import importlib
import pprint
import json

import numpy as np
import cv2
import networkx as nx
import open3d as o3d
from scipy import linalg as la
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm.contrib import tenumerate

import utils

class Tracker:

    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }

    def __init__(self, data_cfg, rod_mesh_file):
        self.data_cfg = data_cfg
        pprint.pprint(self.data_cfg)
        self.initialized = False

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
        
        # read rod mesh for ICP
        rod_mesh = o3d.io.read_triangle_mesh(rod_mesh_file)
        self.rod_pcd = rod_mesh.sample_points_poisson_disk(3000)
        points = np.asarray(self.rod_pcd.points)
        offset = points.mean(0)
        points -= offset  # move point cloud center
        points /= self.data_cfg['rod_scale']  # scale points from millimeter to meter

        # visualize rod and end caps
        init_rod_pose = np.eye(4)
        self.rod_length = self.data_cfg['rod_length']
        rod_frame = utils.generate_coordinate_frame(init_rod_pose)
        end_cap_pose_1 = init_rod_pose.copy()
        end_cap_pose_1[:3, 3] += end_cap_pose_1[:3, 2] * self.rod_length / 2
        end_cap_frame_1 = utils.generate_coordinate_frame(end_cap_pose_1)
        end_cap_pose_2 = init_rod_pose.copy()
        end_cap_pose_2[:3, 3] -= end_cap_pose_2[:3, 2] * self.rod_length / 2
        end_cap_frame_2 = utils.generate_coordinate_frame(end_cap_pose_2)
        o3d.visualization.draw_geometries([self.rod_pcd, rod_frame, end_cap_frame_1, end_cap_frame_2])

    def initialize(self, color_im: np.ndarray, depth_im: np.ndarray, visualize: bool = True, compute_hsv: bool = True):
        scene_pcd = utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im,
                                     depth_trunc=self.data_cfg['depth_trunc'])
        o3d.visualization.draw_geometries([scene_pcd])
        if 'cam_extr' not in self.data_cfg:
            plane_frame, _ = utils.plane_detection_ransac(scene_pcd, inlier_thresh=0.005, visualize=visualize)
            self.data_cfg['cam_extr'] = np.round(plane_frame, decimals=3)
        scene_pcd.transform(la.inv(self.data_cfg['cam_extr']))
        
        if 'init_end_cap_rois' not in self.data_cfg:
            color_im_bgr = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
            end_cap_rois = dict()
            for u in self.G.nodes:
                color = self.G.nodes[u]['color']
                window_name = f'Select #{u} end cap ({color})'
                end_cap_roi = cv2.selectROI(window_name, color_im_bgr, fromCenter=False, showCrosshair=False)
                cv2.destroyWindow(window_name)
                end_cap_rois[u] = end_cap_roi
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
        reconstructed_rods = o3d.geometry.PointCloud()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            end_cap_centers = []
            obs_pcd = o3d.geometry.PointCloud()

            for node in (u, v):
                roi = self.data_cfg['init_end_cap_rois'][str(node)]
                pt1 = (int(roi[0]), int(roi[1]))
                pt2 = (pt1[0] + int(roi[2]), pt1[1] + int(roi[3]))
                masked_depth_im = np.zeros_like(depth_im)
                masked_depth_im[pt1[1]:pt2[1], pt1[0]:pt2[0]] = depth_im[pt1[1]:pt2[1], pt1[0]:pt2[0]]

                color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)
                masked_color_im = np.zeros_like(color_im_hsv)
                masked_color_im[pt1[1]:pt2[1], pt1[0]:pt2[0]] = color_im_hsv[pt1[1]:pt2[1], pt1[0]:pt2[0]]

                end_cap_pcd = utils.create_pcd(masked_depth_im, self.data_cfg['cam_intr'], masked_color_im,
                                               cam_extr=self.data_cfg['cam_extr'])

                # filter end_cap_pcd by HSV
                points_hsv = np.asarray(end_cap_pcd.colors) * 255
                mask1 = np.all((points_hsv - self.data_cfg['hsv_ranges'][color][0]) > 0, axis=1)
                mask2 = np.all((points_hsv - self.data_cfg['hsv_ranges'][color][1]) < 0, axis=1)

                if color == 'red':
                    mask3 = np.all((points_hsv - np.array([0, 150, 50])) > 0, axis=1)
                    mask4 = np.all((points_hsv - np.array([20, 255, 255])) < 0, axis=1)
                    valid_indices = np.where((mask1 & mask2) | (mask3 & mask4))[0]
                else:
                    valid_indices = np.where(mask1 & mask2)[0]

                end_cap_pcd = end_cap_pcd.select_by_index(valid_indices)

                # filter end_cap_pcd by finding the largest cluster
                labels = np.asarray(end_cap_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
                points_for_each_cluster = [(labels == label).sum() for label in range(labels.max() + 1)]
                label = np.argmax(points_for_each_cluster)
                masked_indices = np.where(labels == label)[0]
                end_cap_pcd = end_cap_pcd.select_by_index(masked_indices)

                end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)
                end_cap_centers.append(end_cap_center)

                obs_pcd += end_cap_pcd
            
            # compute rod pose given end cap centers
            init_pose = self.estimate_rod_pose_from_end_cap_centers(end_cap_centers)
            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
            icp_result = utils.icp(rod_pcd, obs_pcd, max_iter=30, init=init_pose)
            print(f"init icp fitness ({color}):", icp_result.fitness)
            rod_pose = icp_result.transformation
            rod_pcd.transform(rod_pose)
            reconstructed_rods += rod_pcd
            self.G.edges[u, v]['pose_list'].append(rod_pose)
            self.G.nodes[u]['pos_list'].append(rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[v]['pos_list'].append(rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2)

        if visualize:
            o3d.visualization.draw_geometries([scene_pcd, reconstructed_rods])
        
        self.initialized = True

    def update(self, color_im, depth_im, info, visualizer=None):
        assert self.initialized, "[Error] You must first initialize the tracker!"
        color_im_vis = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
        color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)        
        scene_pcd_hsv = utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im_hsv,
                                         depth_trunc=self.data_cfg['depth_trunc'],
                                         cam_extr=self.data_cfg['cam_extr'])
        scene_pcd = utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im,
                                     depth_trunc=self.data_cfg['depth_trunc'],
                                     cam_extr=self.data_cfg['cam_extr'])
        self.scene_pcd = scene_pcd

        self.track_with_rgbd(scene_pcd_hsv)
        self.constrained_optimization(info)

        if visualizer is not None:
            visualizer.clear_geometries()
            visualizer.add_geometry(scene_pcd)
            for color, (u, v) in self.data_cfg['color_to_rod'].items():
                rod_pcd = copy.deepcopy(self.rod_pcd)
                rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                rod_pose = self.G.edges[u, v]['pose_list'][-1]
                rod_pcd.transform(rod_pose)
                visualizer.add_geometry(rod_pcd)

            # Due to a potential bug in Open3D, cx, cy can only be w / 2 - 0.5, h / 2 - 0.5
            # https://github.com/intel-isl/Open3D/issues/1164
            cam_intr_vis = o3d.camera.PinholeCameraIntrinsic()
            cam_intr = np.array(self.data_cfg['cam_intr'])
            fx, fy = cam_intr[0, 0], cam_intr[1, 1]
            cx, cy = self.data_cfg['im_w'] / 2 - 0.5, self.data_cfg['im_h'] / 2 - 0.5
            cam_intr_vis.set_intrinsics(self.data_cfg['im_w'], self.data_cfg['im_h'], fx, fy, cx, cy)
            cam_params = o3d.camera.PinholeCameraParameters()
            cam_params.intrinsic = cam_intr_vis
            cam_params.extrinsic = self.data_cfg['cam_extr']

            visualizer.get_view_control().convert_from_pinhole_camera_parameters(cam_params)
            visualizer.poll_events()
            visualizer.update_renderer()

        cv2.imshow("observation", color_im_vis)
        if cv2.waitKey(1) == ord('q'):
            exit(0)

    def track_with_rgbd(self, scene_pcd_hsv):
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            obs_pcd = o3d.geometry.PointCloud()
            prev_pose = self.G.edges[u, v]['pose_list'][-1].copy()

            for node in (u, v):
                prev_end_cap_pos = self.G.nodes[node]['pos_list'][-1]

                # points = np.asarray(scene_pcd_hsv.points)
                # dist = la.norm(points - prev_end_cap_pos[None], axis=1)
                # valid_indices = np.where(dist < 0.04)[0]
                # cropped_cloud = scene_pcd_hsv.select_by_index(valid_indices)

                half_length = 0.04
                lowerbound = prev_end_cap_pos - half_length
                upperbound = prev_end_cap_pos + half_length
                end_cap_bbox = o3d.geometry.AxisAlignedBoundingBox(lowerbound, upperbound)
                cropped_cloud = scene_pcd_hsv.crop(end_cap_bbox)

                points_hsv = np.asarray(cropped_cloud.colors) * 255
                mask1 = np.all((points_hsv - self.data_cfg['hsv_ranges'][color][0]) > 0, axis=1)
                mask2 = np.all((points_hsv - self.data_cfg['hsv_ranges'][color][1]) < 0, axis=1)
                if color == 'red':
                    mask3 = np.all((points_hsv - np.array([0, 100, 50])) > 0, axis=1)
                    mask4 = np.all((points_hsv - np.array([20, 255, 255])) < 0, axis=1)
                    valid_indices = np.where((mask1 & mask2) | (mask3 & mask4))[0]
                else:
                    valid_indices = np.where(mask1 & mask2)[0]

                # get point cloud for ICP refinement
                if len(valid_indices) > 50:  # There are enough points for ICP
                    end_cap_pcd = cropped_cloud.select_by_index(valid_indices)
                    end_cap_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    obs_pcd += end_cap_pcd
                elif len(valid_indices) > 0:  # There are some points but not enough for ICP
                    end_cap_pcd = cropped_cloud.select_by_index(valid_indices)
                    end_cap_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    obs_pcd += end_cap_pcd

                    # randomly sample points around previous end cap center
                    sampled_pts = np.random.normal(loc=prev_end_cap_pos, scale=0.005, size=(100, 3))
                    sampled_pcd = o3d.geometry.PointCloud()
                    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts)
                    sampled_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    obs_pcd += sampled_pcd
                else:  # no point exists, randomly sample points around previous end cap center
                    sampled_pts = np.random.normal(loc=prev_end_cap_pos, scale=0.005, size=(100, 3))
                    sampled_pcd = o3d.geometry.PointCloud()
                    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts)
                    sampled_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    obs_pcd += sampled_pcd

            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)

            icp_result = utils.icp(rod_pcd, obs_pcd, max_iter=30, init=prev_pose)
            if icp_result.fitness > 0.7 and la.norm(icp_result.transformation[:3, 3] - prev_pose[:3, 3]) < 0.05:
                curr_pose = icp_result.transformation
            else:
                curr_pose = prev_pose.copy()

            self.G.edges[u, v]['pose_list'].append(curr_pose)
            self.G.nodes[u]['pos_list'].append(curr_pose[:3, 3] - curr_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[v]['pos_list'].append(curr_pose[:3, 3] + curr_pose[:3, 2] * self.rod_length / 2)

    def constrained_optimization(self, info):
        rod_length = self.data_cfg['rod_length']
        num_end_caps = 2 * self.data_cfg['num_rods']  # a rod has two end caps

        sensor_measurement = info['sensors']
        obj_func = self.objective_function_generator(sensor_measurement, balance_factor=0.5)

        init_values = np.zeros(3 * num_end_caps)
        for i in range(num_end_caps):
            init_values[(3*i):(3*i + 3)] = self.G.nodes[i]['pos_list'][-1]

        rod_constraints = dict()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            constraint = dict()
            constraint['type'] = 'eq'
            constraint['fun'] = lambda x: la.norm(x[(3*u):(3*u + 3)] - x[(3*v):(3*v + 3)]) - rod_length
            rod_constraints[color] = constraint
        rod_constraints = tuple(rod_constraints.values())

        res = minimize(obj_func, init_values, method='SLSQP', constraints=rod_constraints)

        for i in range(num_end_caps):
            self.G.nodes[i]['pos_list'][-1] = res.x[(3*i):(3*i + 3)].copy()
        
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            prev_rod_pose = self.G.edges[u, v]['pose_list'][-1]
            u_pos = self.G.nodes[u]['pos_list'][-1]
            v_pos = self.G.nodes[v]['pos_list'][-1]
            curr_end_cap_centers = np.vstack([u_pos, v_pos])
            optimized_pose = self.estimate_rod_pose_from_end_cap_centers(curr_end_cap_centers, prev_rod_pose)
            self.G.edges[u, v]['pose_list'][-1] = optimized_pose
    
    def objective_function_generator(self, sensor_measurement, balance_factor=0.5):
        def objective_function(X):
            unary_loss = 0
            for i in range(len(self.data_cfg['node_to_color'])):
                pos = X[(3*i):(3*i + 3)]
                estimated_pos = self.G.nodes[i]['pos_list'][-1]
                unary_loss += la.norm(pos - estimated_pos)**2
            binary_loss = 0
            for sensor_id, (u, v) in data_cfg['sensor_to_tendon'].items():
                u_pos = X[(3*u):(3*u + 3)]
                v_pos = X[(3*v):(3*v + 3)]
                estimated_length = la.norm(u_pos - v_pos)
                measured_length = sensor_measurement[str(sensor_id)]['length'] / 100
                binary_loss += (estimated_length - measured_length)**2
            return balance_factor * unary_loss + (1 - balance_factor) * binary_loss
        return objective_function

    def estimate_rod_pose_from_end_cap_centers(self, curr_end_cap_centers, prev_rod_pose=None):
        curr_rod_pos = (curr_end_cap_centers[0] + curr_end_cap_centers[1]) / 2
        curr_z_dir = curr_end_cap_centers[1] - curr_end_cap_centers[0]
        curr_z_dir /= la.norm(curr_z_dir)

        if prev_rod_pose is None:
            prev_rod_pose = np.eye(4)
        
        prev_rot = prev_rod_pose[:3, :3].copy()
        prev_z_dir = prev_rot[:, 2].copy()

        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        delta_rot = utils.np_rotmat_of_two_v(v1=prev_z_dir, v2=curr_z_dir)
        curr_rod_pose = np.eye(4)
        curr_rod_pose[:3, :3] = delta_rot @ prev_rot
        curr_rod_pose[:3, 3] = curr_rod_pos
        return curr_rod_pose

    # def estimate_rod_pose_from_end_cap_centers(self, end_cap_centers):
    #     pos = (end_cap_centers[0] + end_cap_centers[1]) / 2
    #     z_dir = end_cap_centers[1] - end_cap_centers[0]
    #     z_dir /= la.norm(z_dir)
    #     x_dir = np.array([-z_dir[2], 0, z_dir[0]])
    #     x_dir /= la.norm(x_dir)
    #     y_dir = np.cross(z_dir, x_dir)
    #     y_dir /= la.norm(y_dir)

    #     pose = np.eye(4)
    #     pose[:3, 0] = x_dir
    #     pose[:3, 1] = y_dir
    #     pose[:3, 2] = z_dir
    #     pose[:3, 3] = pos
    #     return pose


if __name__ == '__main__':
    dataset = 'dataset'
    video_id = "crawling_trial4"
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(dataset, video_id, 'color'))])

    data_cfg_module = importlib.import_module(f'{dataset}.{video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    rod_mesh_file = 'pcd/yale/untethered_rod_w_end_cap.ply'
    tracker = Tracker(data_cfg, rod_mesh_file=rod_mesh_file)

    # initialize tracker with the first frame    
    color_path = os.path.join(dataset, video_id, 'color', f'{prefixes[0]}.png')
    depth_path = os.path.join(dataset, video_id, 'depth', f'{prefixes[0]}.png')
    color_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
    tracker.initialize(color_im, depth_im, visualize=True, compute_hsv=False)
    data_cfg_module.write_config(tracker.data_cfg)

    # track frames
    os.makedirs(os.path.join(dataset, video_id, "scene_cloud"), exist_ok=True)
    os.makedirs(os.path.join(dataset, video_id, "estimation_cloud"), exist_ok=True)
    os.makedirs(os.path.join(dataset, video_id, "raw_estimation"), exist_ok=True)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=data_cfg['im_w'], height=data_cfg['im_h'])
    for idx, prefix in tenumerate(prefixes[1:]):
        color_path = os.path.join(dataset, video_id, 'color', f'{prefix}.png')
        depth_path = os.path.join(dataset, video_id, 'depth', f'{prefix}.png')
        info_path = os.path.join(dataset, video_id, 'data', f'{prefix}.json')

        color_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
        with open(info_path, 'r') as f:
            info = json.load(f)

        tracker.update(color_im, depth_im, info, visualizer=visualizer)
        # o3d.io.write_point_cloud(os.path.join(dataset, video_id, "scene_cloud", f"{idx:04d}.ply"), tracker.scene_pcd)
        # o3d.io.write_point_cloud(os.path.join(dataset, video_id, "estimation_cloud", f"{idx:04d}.ply"), tracker.estimation_cloud)
        visualizer.capture_screen_image(os.path.join(dataset, video_id, "raw_estimation", f"{idx:04d}.png"))

    # save rod poses and end cap positions to file
    pose_output_folder = os.path.join(dataset, video_id, "poses")
    os.makedirs(pose_output_folder, exist_ok=True)
    for color, (u, v) in tracker.data_cfg['color_to_rod'].items():
        np.save(os.path.join(pose_output_folder, f'{color}.npy'), np.array(tracker.G.edges[u, v]['pose_list']))
        np.save(os.path.join(pose_output_folder, f'{u}_pos.npy'), np.array(tracker.G.nodes[u]['pos_list']))
        np.save(os.path.join(pose_output_folder, f'{v}_pos.npy'), np.array(tracker.G.nodes[v]['pos_list']))
