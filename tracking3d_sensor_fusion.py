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
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm

import perception_utils


class Tracker:

    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }


    def __init__(self, data_cfg, rod_pcd):
        self.data_cfg = data_cfg
        pprint.pprint(self.data_cfg)
        self.initialized = False

        self.rod_pcd = copy.deepcopy(rod_pcd)
        self.rod_length = self.data_cfg['rod_length']

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
        complete_obs_pcd = o3d.geometry.PointCloud()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            print("color: ", color)
            print(self.data_cfg['hsv_ranges'][color])
            end_cap_centers = []
            obs_pcd = o3d.geometry.PointCloud()

            for node in (u, v):
                roi = self.data_cfg['init_end_cap_rois'][str(node)]
                pt1 = (int(roi[0]), int(roi[1]))
                pt2 = (pt1[0] + int(roi[2]), pt1[1] + int(roi[3]))
                masked_depth_im = np.zeros_like(depth_im)
                masked_depth_im[pt1[1]:pt2[1], pt1[0]:pt2[0]] = depth_im[pt1[1]:pt2[1], pt1[0]:pt2[0]]

                color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)
                masked_hsv_im = np.zeros_like(color_im_hsv)
                masked_hsv_im[pt1[1]:pt2[1], pt1[0]:pt2[0]] = color_im_hsv[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                end_cap_pcd = perception_utils.create_pcd(masked_depth_im, self.data_cfg['cam_intr'], masked_hsv_im,
                                                          cam_extr=self.data_cfg['cam_extr'])
                # o3d.visualization.draw_geometries([end_cap_pcd])

                # filter end_cap_pcd by HSV
                points_hsv = np.asarray(end_cap_pcd.colors) * 255
                mask1 = np.all((points_hsv - self.data_cfg['hsv_ranges'][color][0]) >= 0, axis=1)
                mask2 = np.all((points_hsv - self.data_cfg['hsv_ranges'][color][1]) <= 0, axis=1)

                if color == 'red':
                    mask3 = np.all((points_hsv - np.array([0, 50, 50])) >= 0, axis=1)
                    mask4 = np.all((points_hsv - np.array([20, 255, 255])) <= 0, axis=1)
                    valid_indices = np.where((mask1 & mask2) | (mask3 & mask4))[0]
                else:
                    valid_indices = np.where(mask1 & mask2)[0]

                end_cap_pcd = end_cap_pcd.select_by_index(valid_indices)

                end_cap_pcd = end_cap_pcd.paint_uniform_color(np.array(self.ColorDict[color]) / 255)
                # o3d.visualization.draw_geometries([end_cap_pcd])

                # filter end_cap_pcd by finding the largest cluster
                labels = np.asarray(end_cap_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
                points_for_each_cluster = [(labels == label).sum() for label in range(labels.max() + 1)]
                label = np.argmax(points_for_each_cluster)
                masked_indices = np.where(labels == label)[0]
                end_cap_pcd = end_cap_pcd.select_by_index(masked_indices)
                # o3d.visualization.draw_geometries([end_cap_pcd])

                end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)
                end_cap_centers.append(end_cap_center)

                obs_pcd += end_cap_pcd

            complete_obs_pcd += obs_pcd

            # compute rod pose given end cap centers
            init_pose = self.estimate_rod_pose_from_end_cap_centers(end_cap_centers)
            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
            icp_result = perception_utils.icp(rod_pcd, obs_pcd, max_iter=30, init=init_pose)
            print(f"init icp fitness ({color}):", icp_result.fitness)
            rod_pose = icp_result.transformation
            self.G.edges[u, v]['pose_list'].append(rod_pose)
            self.G.nodes[u]['pos_list'].append(rod_pose[:3, 3] + rod_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[v]['pos_list'].append(rod_pose[:3, 3] - rod_pose[:3, 2] * self.rod_length / 2)

        # o3d.visualization.draw_geometries([complete_obs_pcd])

        self.constrained_optimization(info, balance_factor=0.8)

        robot_cloud = o3d.geometry.PointCloud()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
            rod_pose = self.G.edges[u, v]['pose_list'][-1]
            rod_pcd.transform(rod_pose)
            robot_cloud += rod_pcd

        robot_cloud = self.rigid_finetune(complete_obs_pcd, robot_cloud)
        if visualize:
            o3d.visualization.draw_geometries([scene_pcd, robot_cloud])
        self.initialized = True


    def update(self, color_im, depth_im, info):
        assert self.initialized, "[Error] You must first initialize the tracker!"
        color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)
        scene_pcd_hsv = perception_utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im_hsv,
                                                    depth_trunc=self.data_cfg['depth_trunc'],
                                                    cam_extr=self.data_cfg['cam_extr'])
        scene_pcd = perception_utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im,
                                                depth_trunc=self.data_cfg['depth_trunc'],
                                                cam_extr=self.data_cfg['cam_extr'])

        complete_obs_pcd = self.track_with_rgbd(scene_pcd_hsv)
        self.constrained_optimization(info, balance_factor=0.8)

        robot_cloud = o3d.geometry.PointCloud()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
            rod_pose = self.G.edges[u, v]['pose_list'][-1]
            rod_pcd.transform(rod_pose)
            robot_cloud += rod_pcd

        robot_cloud = self.rigid_finetune(complete_obs_pcd, robot_cloud)
        return robot_cloud, scene_pcd


    def visualize(self, robot_cloud, scene_pcd, visualizer):
        visualizer.clear_geometries()
        visualizer.add_geometry(scene_pcd)
        visualizer.add_geometry(robot_cloud)

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

        error_after = np.mean(la.norm(Q - (R @ P + t), axis=0))

        cam2mc_mat = np.zeros((4, 4))
        cam2mc_mat[0:3, 0:3] = R
        cam2mc_mat[0:3, 3] = t.squeeze()
        print(cam2mc_mat)
        return cam2mc_mat


    def track_with_rgbd(self, scene_pcd_hsv):
        complete_obs_pcd = o3d.geometry.PointCloud()
        for color, (u, v) in self.data_cfg['color_to_rod'].items():
            end_cap_obs_cloud = o3d.geometry.PointCloud()
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
                mask1 = np.all((points_hsv - self.data_cfg['hsv_ranges'][color][0]) >= 0, axis=1)
                mask2 = np.all((points_hsv - self.data_cfg['hsv_ranges'][color][1]) <= 0, axis=1)
                if color == 'red':
                    mask3 = np.all((points_hsv - np.array([0, 50, 50])) >= 0, axis=1)
                    mask4 = np.all((points_hsv - np.array([20, 255, 255])) <= 0, axis=1)
                    valid_indices = np.where((mask1 & mask2) | (mask3 & mask4))[0]
                else:
                    valid_indices = np.where(mask1 & mask2)[0]

                # get point cloud for ICP refinement
                if len(valid_indices) > 50:  # There are enough points for ICP
                    end_cap_pcd = cropped_cloud.select_by_index(valid_indices)
                    end_cap_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    end_cap_obs_cloud += end_cap_pcd
                elif len(valid_indices) > 0:  # There are some points but not enough for ICP
                    end_cap_pcd = cropped_cloud.select_by_index(valid_indices)
                    end_cap_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    end_cap_obs_cloud += end_cap_pcd

                    # randomly sample points around previous end cap center
                    sampled_pts = np.random.normal(loc=prev_end_cap_pos, scale=0.005, size=(100, 3))
                    sampled_pcd = o3d.geometry.PointCloud()
                    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts)
                    sampled_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    end_cap_obs_cloud += sampled_pcd
                else:  # no point exists, randomly sample points around previous end cap center
                    sampled_pts = np.random.normal(loc=prev_end_cap_pos, scale=0.005, size=(100, 3))
                    sampled_pcd = o3d.geometry.PointCloud()
                    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts)
                    sampled_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    end_cap_obs_cloud += sampled_pcd

                if len(valid_indices) > 0:
                    complete_obs_pcd += end_cap_pcd

            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)

            icp_result = perception_utils.icp(rod_pcd, end_cap_obs_cloud, max_iter=30, init=prev_pose)
            if icp_result.fitness > 0.7 and la.norm(icp_result.transformation[:3, 3] - prev_pose[:3, 3]) < 0.05:
                curr_pose = icp_result.transformation
            else:
                curr_pose = prev_pose.copy()

            self.G.edges[u, v]['pose_list'].append(curr_pose)
            self.G.nodes[u]['pos_list'].append(curr_pose[:3, 3] + curr_pose[:3, 2] * self.rod_length / 2)
            self.G.nodes[v]['pos_list'].append(curr_pose[:3, 3] - curr_pose[:3, 2] * self.rod_length / 2)

        return complete_obs_pcd


    def constrained_optimization(self, info, balance_factor=0.8):
        rod_length = self.data_cfg['rod_length']
        num_end_caps = 2 * self.data_cfg['num_rods']  # a rod has two end caps

        sensor_measurement = info['sensors']
        sensor_status = info['sensor_status']
        obj_func = self.objective_function_generator(sensor_measurement, sensor_status, balance_factor=balance_factor)

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


    def objective_function_generator(self, sensor_measurement, sensor_status, balance_factor=0.5):

        def objective_function(X):
            unary_loss = 0
            for i in range(len(self.data_cfg['node_to_color'])):
                pos = X[(3 * i):(3 * i + 3)]
                estimated_pos = self.G.nodes[i]['pos_list'][-1]
                unary_loss += np.sum((pos - estimated_pos)**2)
                # unary_loss += la.norm(pos - estimated_pos)

            binary_loss = 0
            for sensor_id, (u, v) in data_cfg['sensor_to_tendon'].items():
                if not sensor_status[sensor_id]:  # sensor is broken
                    continue

                u_pos = X[(3 * u):(3 * u + 3)]
                v_pos = X[(3 * v):(3 * v + 3)]
                estimated_length = la.norm(u_pos - v_pos)
                measured_length = sensor_measurement[str(sensor_id)]['length'] / 100
                binary_loss += (estimated_length - measured_length)**2
                # binary_loss += np.abs(estimated_length - measured_length)

            return balance_factor * unary_loss + (1 - balance_factor) * binary_loss

        return objective_function


    def constraint_function_generator(self, u, v, rod_length):

        def constraint_function(X):
            u_pos = X[3*u : 3*u + 3]
            v_pos = X[3*v : 3*v + 3]
            return la.norm(u_pos - v_pos) - rod_length
            # return (la.norm(u_pos - v_pos) - rod_length)**2

        return constraint_function


    def rigid_finetune(self, obs_cloud, robot_cloud):
        icp_result = perception_utils.icp(robot_cloud, obs_cloud, max_iter=30, init=np.eye(4))
        if icp_result.fitness > 0.7 and la.norm(icp_result.transformation[:3, 3]) < 0.05:
            delta_transformation = icp_result.transformation
        else:
            delta_transformation = np.eye(4)

        for _, (u, v) in self.data_cfg['color_to_rod'].items():
            prev_rod_pose = self.G.edges[u, v]['pose_list'][-1]
            optimized_pose = delta_transformation @ prev_rod_pose
            self.G.edges[u, v]['pose_list'][-1] = optimized_pose
            self.G.nodes[u]['pos_list'][-1] = optimized_pose[:3, 3] + optimized_pose[:3, 2] * self.rod_length / 2
            self.G.nodes[v]['pos_list'][-1] = optimized_pose[:3, 3] - optimized_pose[:3, 2] * self.rod_length / 2

        robot_cloud.transform(delta_transformation)
        return robot_cloud


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
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    # parser.add_argument("--video_id", default="fabric2")
    # parser.add_argument("--video_id", default="six_cameras10")
    # parser.add_argument("--video_id", default="crawling_sim")
    parser.add_argument("--video_id", default="socks6")
    parser.add_argument("--rod_mesh_file", default="pcd/yale/untethered_rod_w_end_cap.ply")
    parser.add_argument("--rod_pcd_file", default="pcd/yale/untethered_rod_w_end_cap.pcd")
    # parser.add_argument("--first_frame_id", default=0, type=int)
    parser.add_argument("--first_frame_id", default=50, type=int)
    parser.add_argument("-v", "--visualize", default=True, action="store_true")
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video_id)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'color'))])

    # read data config
    data_cfg_module = importlib.import_module(f'{args.dataset}.{args.video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    # read rod geometry
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

    tracker = Tracker(data_cfg, rod_pcd)

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

        robot_cloud, scene_pcd = tracker.update(color_im, depth_im, info)
        estimation_cloud = robot_cloud + scene_pcd

        tracker.compute_mc2cam_mat(info)

        if args.visualize:
            tracker.visualize(robot_cloud, scene_pcd, visualizer)
            cv2.imshow("observation", color_im_bgr)
            cv2.waitKey(1)

        o3d.io.write_point_cloud(os.path.join(video_path, "estimation_cloud", f"{idx:04d}.ply"), estimation_cloud)
        visualizer.capture_screen_image(os.path.join(video_path, "raw_estimation", f"{idx:04d}.png"))

    # save rod poses and end cap positions to file
    pose_output_folder = os.path.join(video_path, "poses")
    os.makedirs(pose_output_folder, exist_ok=True)
    for color, (u, v) in tracker.data_cfg['color_to_rod'].items():
        np.save(os.path.join(pose_output_folder, f'{color}.npy'), np.array(tracker.G.edges[u, v]['pose_list']))
        np.save(os.path.join(pose_output_folder, f'{u}_pos.npy'), np.array(tracker.G.nodes[u]['pos_list']))
        np.save(os.path.join(pose_output_folder, f'{v}_pos.npy'), np.array(tracker.G.nodes[v]['pos_list']))
