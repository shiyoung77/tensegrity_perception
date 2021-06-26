import os
import time
import copy
import importlib
import pprint
import open3d as o3d
import numpy as np
from scipy import linalg as la
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.contrib import tenumerate

import utils

class Tracker:

    ColorDict = {
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255]
    }

    def __init__(self, data_cfg, rod_mesh_file, tracker2D='csrt'):
        self.data_cfg = data_cfg
        self.initialized = False
        self.trackers = dict()
        self.rois = dict()
        self.end_cap_centers = dict()
        self.hsv_ranges = dict()
        self.pose_results = {color: [] for color in self.data_cfg['end_cap_colors']}

        # read rod mesh
        self.rod_pcd = o3d.io.read_point_cloud(rod_mesh_file)
        points = np.asarray(self.rod_pcd.points)
        points -= points.mean(0)  # move point cloud center
        points /= 1000  # scale points from millimeter to meter


    def initialize(self, color_im: np.ndarray, depth_im: np.ndarray, visualize: bool = True, compute_hsv: bool = True):
        scene_pcd = utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im, depth_trunc=0.6)
        if 'cam_extr' not in self.data_cfg:
            plane_frame, _ = utils.plane_detection_ransac(scene_pcd, inlier_thresh=0.005, visualize=visualize)
            self.data_cfg['cam_extr'] = np.round(plane_frame, decimals=3)
        scene_pcd.transform(la.inv(self.data_cfg['cam_extr']))
        
        if 'init_end_cap_rois' not in self.data_cfg:
            color_im_bgr = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
            end_cap_rois = dict()
            for color in self.data_cfg['end_cap_colors']:
                rois = cv2.selectROIs(f"select {color} cap", color_im_bgr, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow(f"select {color} cap")
                end_cap_rois[color] = rois.tolist()
            self.data_cfg['init_end_cap_rois'] = end_cap_rois

        # visualize init bboxes
        color_im_vis = color_im.copy()
        for color, rois in self.data_cfg['init_end_cap_rois'].items():
            for roi in rois:
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
            for color, rois in self.data_cfg['init_end_cap_rois'].items():
                h_hist = np.zeros((256, 1))
                s_hist = np.zeros((256, 1))
                v_hist = np.zeros((256, 1))
                for i, roi in enumerate(rois):
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
        for color, rois in self.data_cfg['init_end_cap_rois'].items():
            # compute 3D centers of each end cap
            end_cap_centers = []
            obs_pcd = o3d.geometry.PointCloud()
            for i, roi in enumerate(rois):
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

                obs_pcd += end_cap_pcd
                end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)
                end_cap_centers.append(end_cap_center)
                self.end_cap_centers.setdefault(color + '-' + str(i), []).append(end_cap_center)
            
            # compute rod pose given end cap centers
            init_pose = self.estimate_rod_pos_from_end_cap_centers(end_cap_centers)
            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
            icp_result = utils.icp(rod_pcd, obs_pcd, max_iter=30, init=init_pose)
            print(f"init icp fitness ({color}):", icp_result.fitness)
            rod_pcd.transform(icp_result.transformation)
            reconstructed_rods += rod_pcd
            self.pose_results[color].append(icp_result.transformation)

        if visualize:
            vis_list = [scene_pcd, reconstructed_rods]
            for _, end_cap_center_list in self.end_cap_centers.items():
                half_length = 0.025
                lowerbound = np.array(end_cap_center_list[-1]) - half_length
                upperbound = np.array(end_cap_center_list[-1]) + half_length
                vis_list.append(o3d.geometry.AxisAlignedBoundingBox(lowerbound, upperbound))
            o3d.visualization.draw_geometries(vis_list)
        
        self.initialized = True


    def update(self, color_im, depth_im, visualizer=None):
        assert self.initialized, "[Error] You must initialize the tracker!"
        color_im_vis = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)

        color_im_hsv = cv2.cvtColor(color_im, cv2.COLOR_RGB2HSV)        
        scene_pcd_hsv = utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im_hsv, depth_trunc=0.6,
                                         cam_extr=self.data_cfg['cam_extr'])
        box_to_filter_table = o3d.geometry.AxisAlignedBoundingBox([-10, -10, 0.01], [10, 10, 0.3])
        scene_pcd_hsv = scene_pcd_hsv.crop(box_to_filter_table)
        
        scene_pcd = utils.create_pcd(depth_im, self.data_cfg['cam_intr'], color_im, depth_trunc=0.6,
                                     cam_extr=self.data_cfg['cam_extr'])
        # vis_list = []
        vis_list = [scene_pcd]
        reconstructed_rods = o3d.geometry.PointCloud()
        for color in self.data_cfg['end_cap_colors']:
            obs_pcd = o3d.geometry.PointCloud()
            # Start from the front end cap to the back end cap
            end_cap_centers = []
            for i in range(2):
                end_cap_name = color + '-' + str(i)
                prev_end_cap_center = self.end_cap_centers[end_cap_name][-1]

                half_length = 0.025
                lowerbound = np.array(prev_end_cap_center) - half_length
                upperbound = np.array(prev_end_cap_center) + half_length
                end_cap_bbox = o3d.geometry.AxisAlignedBoundingBox(lowerbound, upperbound)
                vis_list.append(end_cap_bbox)

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

                if i == 0:  # assuming the first end cap has better visibility
                    if len(valid_indices) > 50:  # There are enough points for ICP
                        end_cap_pcd = cropped_cloud.select_by_index(valid_indices)
                        end_cap_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                        obs_pcd += end_cap_pcd

                        end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)
                        self.end_cap_centers[end_cap_name].append(end_cap_center)
                    elif len(valid_indices) > 10:  # There are some points but not enough for ICP
                        end_cap_pcd = cropped_cloud.select_by_index(valid_indices)
                        end_cap_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                        end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)

                        self.end_cap_centers[end_cap_name].append(end_cap_center)
                        obs_pcd += end_cap_pcd

                        # randomly sample points around previous center
                        std = 0.005
                        N = 50
                        sampled_pts = np.random.normal(loc=end_cap_center, scale=std, size=(100, 3))
                        sampled_pcd = o3d.geometry.PointCloud()
                        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts)
                        sampled_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                        obs_pcd += sampled_pcd
                    else:  # no point exists
                        end_cap_center = self.end_cap_centers[end_cap_name][-1].copy() # reuse the previous end cap center
                        self.end_cap_centers[end_cap_name].append(end_cap_center)

                        # randomly sample points around previous center
                        std = 0.005
                        N = 100
                        sampled_pts = np.random.normal(loc=end_cap_center, scale=std, size=(100, 3))
                        sampled_pcd = o3d.geometry.PointCloud()
                        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts)
                        sampled_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                        obs_pcd += sampled_pcd
                    end_cap_centers.append(end_cap_center)
                else:  # assuming the second end cap has worse visibility
                    if len(valid_indices) > 50:  # There are enough points for ICP
                        end_cap_pcd = cropped_cloud.select_by_index(valid_indices)
                        end_cap_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                        obs_pcd += end_cap_pcd

                        end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)
                        self.end_cap_centers[end_cap_name].append(end_cap_center)
                    # elif len(valid_indices) > 0:  # There are some points but not enough for ICP
                    #     end_cap_pcd = cropped_cloud.select_by_index(valid_indices)
                    #     end_cap_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    #     end_cap_center = np.asarray(end_cap_pcd.points).mean(axis=0)

                    #     self.end_cap_centers[end_cap_name].append(end_cap_center)
                    #     obs_pcd += end_cap_pcd

                    #     # randomly sample points around previous center
                    #     std = 0.005
                    #     N = 200
                    #     sampled_pts = np.random.normal(loc=end_cap_center, scale=std, size=(100, 3))
                    #     sampled_pcd = o3d.geometry.PointCloud()
                    #     sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts)
                    #     sampled_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                    #     obs_pcd += sampled_pcd
                    else:
                        fc = self.end_cap_centers[color + '-0'][-1]  # first end cap center
                        cam_intr = np.array(self.data_cfg['cam_intr'])
                        cam_extr = np.array(self.data_cfg['cam_extr'])
                        fc_cam = (cam_extr @ np.array([fc[0], fc[1], fc[2], 1]))[:3]
                        x1, y1, z1 = fc_cam

                        sc = self.end_cap_centers[color + '-1'][-1]  # second end cap center
                        sc_cam = (cam_extr @ np.array([sc[0], sc[1], sc[2], 1]))[:3]

                        fx = cam_intr[0, 0]
                        fy = cam_intr[1, 1]
                        cx = cam_intr[0, 2]
                        cy = cam_intr[1, 2]

                        x = int(round(fx * sc_cam[0] / sc_cam[2] + cx))
                        y = int(round(fy * sc_cam[1] / sc_cam[2] + cy))

                        pt1 = (max(0, x - 30), min(color_im.shape[1], y - 30))
                        pt2 = (max(0, x + 30), min(color_im.shape[0], y + 30))
                        # cv2.rectangle(color_im_vis, pt1, pt2, Tracker.ColorDict[color], 2)

                        
                        mask = cv2.inRange(color_im_hsv, tuple(self.data_cfg['hsv_ranges'][color][0]),
                                                        tuple(self.data_cfg['hsv_ranges'][color][1]))
                        mask = mask.astype(bool)
                        if color == 'red':
                            mask2 = cv2.inRange(color_im_hsv, (0, 100, 50), (20, 255, 255), 2)
                            mask = mask | mask2.astype(bool)
                        color_im_vis[mask] = 0

                        ys, xs = np.where(mask[pt1[1]:pt2[1], pt1[0]:pt2[0]])
                        if len(ys) > 10:
                            x_bar = int(np.mean(xs) + x - 30)
                            y_bar = int(np.mean(ys) + y - 30)

                            pt1 = (max(0, x_bar - 30), min(color_im.shape[1], y_bar - 30))
                            pt2 = (max(0, x_bar + 30), min(color_im.shape[0], y_bar + 30))
                            cv2.rectangle(color_im_vis, pt1, pt2, Tracker.ColorDict[color], 2)
                        else:
                            pc = self.end_cap_centers[end_cap_name][-1]
                            X_prev, Y_prev, Z_prev = (cam_extr @ np.array([pc[0], pc[1], pc[2], 1]))[:3]
                            x_bar = int(round(fx * X_prev / Z_prev + cx))
                            y_bar = int(round(fy * Y_prev / Z_prev + cy))

                        L = 0.21   # rod length
                        A = (x_bar - cx) / fx
                        B = (y_bar - cy) / fy
                        a = A**2 + B**2 + 1
                        b = -2*(A*x1 + B*y1 + z1)
                        c = x1**2 + y1**2 + z1**2 - L**2
                        Z_cam = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                        X_cam = A*Z_cam
                        Y_cam = B*Z_cam

                        end_cap_center = (la.inv(cam_extr) @ np.array([X_cam, Y_cam, Z_cam, 1]))[:3]
                        self.end_cap_centers[end_cap_name].append(end_cap_center)

                        # randomly sample points around previous center
                        std = 0.005
                        N = 200
                        sampled_pts = np.random.normal(loc=end_cap_center, scale=std, size=(100, 3))
                        sampled_pcd = o3d.geometry.PointCloud()
                        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts)
                        sampled_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
                        obs_pcd += sampled_pcd

                        lowerbound = end_cap_center - 0.025
                        upperbound = end_cap_center + 0.025
                        end_cap_bbox = o3d.geometry.AxisAlignedBoundingBox(lowerbound, upperbound)
                        end_cap_bbox.color = np.array([1, 0, 0])
                        vis_list.append(end_cap_bbox)
                    end_cap_centers.append(end_cap_center)

            rod_pcd = copy.deepcopy(self.rod_pcd)
            rod_pcd = rod_pcd.paint_uniform_color(np.array(Tracker.ColorDict[color]) / 255)
            prev_pose = self.pose_results[color][-1].copy()
            rod_pose = self.pose_results[color][-1]  # initialize with previous rod_pose

            icp_result = utils.icp(rod_pcd, obs_pcd, max_iter=30, init=rod_pose)
            if icp_result.fitness > 0.7:
                rod_pose = icp_result.transformation

            if la.norm(rod_pose[:3, 3] - prev_pose[:3, 3]) > 0.05:
                rod_pose = prev_pose.copy()

            rod_pcd.transform(rod_pose)
            self.pose_results[color].append(rod_pose)
            reconstructed_rods += rod_pcd

            vis_list.append(obs_pcd)
            vis_list.append(reconstructed_rods)

        if visualizer is not None:
            visualizer.clear_geometries()
            for geometry in vis_list:
                visualizer.add_geometry(geometry)

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
        cv2.waitKey(1)

    def estimate_rod_pos_from_end_cap_centers(self, end_cap_centers):
        rod_center = (end_cap_centers[0] + end_cap_centers[1]) / 2
        rod_direction = end_cap_centers[0] - end_cap_centers[1]
        rod_direction = rod_direction / np.linalg.norm(rod_direction)
        mesh_frame_principle = np.asarray([0, 0, 1])
        rot_mat = utils.np_rotmat_of_two_v(v1=mesh_frame_principle, v2=rod_direction)
        init_transform_mat = np.eye(4)
        init_transform_mat[:3, :3] = rot_mat
        init_transform_mat[:3, 3] = rod_center
        return init_transform_mat


if __name__ == '__main__':
    dataset = 'dataset'
    video_id = 'no_hooks0'
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(dataset, video_id, 'color'))])

    data_cfg_module = importlib.import_module(f'{dataset}.{video_id}.config')
    data_cfg = data_cfg_module.get_config(read_cfg=True)

    rod_mesh_file = 'pcd/yale/end_cap_with_rod.pcd'
    tracker = Tracker(data_cfg, rod_mesh_file=rod_mesh_file, tracker2D='csrt')

    # initialize tracker with the first frame    
    color_path = os.path.join(dataset, video_id, 'color', f'{prefixes[0]}.png')
    depth_path = os.path.join(dataset, video_id, 'depth', f'{prefixes[0]}.png')
    color_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
    tracker.initialize(color_im, depth_im, visualize=True, compute_hsv=False)
    pprint.pprint(tracker.data_cfg)
    data_cfg_module.write_config(tracker.data_cfg)

    # track frames
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=data_cfg['im_w'], height=data_cfg['im_h'])
    for idx, prefix in tenumerate(prefixes[1:]):
        color_path = os.path.join(dataset, video_id, 'color', f'{prefix}.png')
        depth_path = os.path.join(dataset, video_id, 'depth', f'{prefix}.png')
        color_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / data_cfg['depth_scale']
        tracker.update(color_im, depth_im, visualizer=visualizer)
