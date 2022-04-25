import importlib
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
import scipy.linalg as la

import numpy as np
import pandas as pd
from tqdm import tqdm


class MotionCapture:
    def __init__(self, dataset_name, length_scale=1.):
        self.dataset_name = dataset_name
        self.dataset_folder = Path('./dataset/{}'.format(dataset_name))
        self.length_scale = length_scale
        self.mc_cable_lengths = None
        self.cable_lengths = None
        self.estimated_cable_lengths = None
        self.rgbd_rods_poses = None
        self.cable_ends = None
        self.cable_mapping_to_mujoco = None
        self.motor_direction = None  # motor running forward (1) or backward (-1).
        self.motor_position = None  # current motor position between [0, 1] ([full contraction, full extension])
        self.motor_cmd_speed = None  # commanded motor speed between [0, 99]
        self.motor_cmd_position = None  # commanded target between [0, 1] ([full contraction, full extension])
        self.pe_end_cap_pos_world = None
        self.pe_centered_end_cap_positions = None
        self.has_full_end_cap_pos = None
        self.mc_time_stamps = None
        self.mc_end_cap_pos_obs = None
        self.mc_selected_time_indices = None
        self.mc_selected_end_cap_positions = None
        self.mc_end_cap_pos_world = None
        self.mc_centerd_end_cap_pos_world = None
        self.end_cap_pos_world = None
        self.centered_end_cap_pos_world = None
        self.rod_length = 0.325
        self.mass_r = 0.0175
        self.rod_end_ids = [[0, 1], [2, 3], [4, 5]]
        self.n_tendons = 9
        self.n_rods = 3
        self.n_masses = 6

        # read data config
        spec = importlib.util.spec_from_file_location("module.name", self.dataset_folder / 'config.py')
        data_cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_cfg_module)
        data_cfg = data_cfg_module.get_config(read_cfg=True)
        self.cfg = dict()
        self.cfg['node_to_color'] = data_cfg['node_to_color']
        self.cfg['color_to_rod'] = data_cfg['color_to_rod']
        self.cfg['cam_extr'] = data_cfg['cam_extr']
        cam_to_mocap_filepath = self.dataset_folder / "cam_to_mocap.npy"
        assert cam_to_mocap_filepath.is_file() is True
        self.cfg['mc2cam'] = np.linalg.inv(np.load(cam_to_mocap_filepath))
        self.compute_mc_extrinsic()

        self.read_json_files()

        self.transform_end_cap_positions_from_motion_capture()
        self.transform_end_cap_positions_from_pose_estimation()
        self.set_feet_onto_the_ground()

        self.center_mc_pe_end_cap_positions()
        self.end_cap_pos_world = self.mc_end_cap_pos_world
        self.centered_end_cap_pos_world = self.mc_centerd_end_cap_pos_world

    def compute_cable_lengths(self, end_cap_pos_world, end_cap_pos_mask):
        assert self.T == len(end_cap_pos_world) == len(end_cap_pos_mask)
        cable_lengths_list = list()
        cable_mask_list = list()
        for t in range(self.T):
            cable_lengths = [None for _ in range(self.n_tendons)]
            cable_mask = np.ones((self.n_tendons), dtype=bool)
            end_cap_mask = end_cap_pos_mask[t]
            for cable_id, (end1, end2) in enumerate(self.cable_ends):
                if end_cap_mask[end1] and end_cap_mask[end2]:
                    pos1 = end_cap_pos_world[t, end1]
                    pos2 = end_cap_pos_world[t, end2]
                    diff = np.linalg.norm(pos2 - pos1)
                    cable_lengths[cable_id] = diff
                else:
                    cable_mask[cable_id] = False
            cable_lengths_list.append(cable_lengths)

        self.estimated_cable_lengths = cable_lengths_list  # m
        print(
            f"T=0, estimated cable lengths {self.estimated_cable_lengths[0]}, mc cable lengths {self.mc_cable_lengths[0]}")
        # print(self.estimated_cable_lengths.shape)

    def read_json_files(self):
        data_folder = self.dataset_folder / 'data'
        json_files = list(sorted(data_folder.glob('*.json')))
        mocap_statistics = defaultdict(int)
        n_time_steps = len(json_files)
        self.mc_cable_lengths = list()
        self.motor_direction = list()  # motor running forward (1) or backward (-1).
        self.motor_position = list()  # current motor position between [0, 1] ([full contraction, full extension])
        self.motor_cmd_speed = list()  # commanded motor speed between [0, 99]
        self.motor_cmd_position = list()  # commanded target between [0, 1] ([full contraction, full extension])
        # self.mc_selected_end_cap_positions = list()
        # self.mc_selected_time_indices = list()

        self.mc_time_stamps = list()
        self.has_full_end_cap_pos = list()
        self.mc_end_cap_pos_obs_mask = list()
        self.mc_end_cap_pos_obs = list()
        # self.mc_full_end_cap_indices = list()

        n_full = 0
        for json_file in json_files:
            with open(json_file) as f:
                sensor_data = json.load(f)

                self.mc_time_stamps.append(sensor_data['header']['secs'])

                # motion caption statistics
                has_all = True
                pos_list = list()
                pos_mask = list()
                for i, pos in sensor_data['mocap'].items():
                    if pos is not None:
                        mocap_statistics[i] += 1
                        pos_list.append([pos['x'], pos['y'], pos['z']])
                        pos_mask.append(True)
                    else:
                        has_all = False
                        pos_list.append([0., 0., 0.])
                        pos_mask.append(False)
                if has_all:
                    mocap_statistics['all'] += 1
                self.mc_end_cap_pos_obs.append(pos_list)
                self.mc_end_cap_pos_obs_mask.append(pos_mask)
                self.has_full_end_cap_pos.append(has_all)

                n_cable = len(sensor_data['sensors'])
                self.mc_cable_lengths.append([0.] * n_cable)

                # cable lengths
                for cable_id, cable_sensor in sensor_data['sensors'].items():
                    self.mc_cable_lengths[-1][int(cable_id)] = cable_sensor['length']

                # imu, ignore it

                # motors
                n_motor = len(sensor_data['motors'])
                self.motor_cmd_position.append([0.] * n_motor)
                self.motor_cmd_speed.append([0.] * n_motor)
                self.motor_direction.append([0.] * n_motor)
                self.motor_position.append([0.] * n_motor)
                for motor_id, motor_sensor in sensor_data['motors'].items():
                    motor_id = int(motor_id)
                    if motor_sensor['target'] == 0:
                        self.motor_cmd_position[-1][motor_id] = 0
                    elif motor_sensor['target'] == 1:
                        self.motor_cmd_position[-1][motor_id] = 1
                    self.motor_cmd_speed[-1][motor_id] = motor_sensor['speed'] / 100.
                    if 'forward' in motor_sensor and motor_sensor['forward']:
                        self.motor_direction[-1][motor_id] = 1
                    else:
                        self.motor_direction[-1][motor_id] = -1
                    self.motor_position[-1][motor_id] = motor_sensor['position']

        self.mc_cable_lengths = np.array(self.mc_cable_lengths) / 1000.  # mm to m
        self.motor_cmd_speed = np.array(self.motor_cmd_speed)
        self.motor_cmd_position = np.array(self.motor_cmd_position)
        self.motor_direction = np.array(self.motor_direction)
        self.motor_position = np.array(self.motor_position)
        self.mc_time_stamps = np.array(self.mc_time_stamps)
        self.mc_time_stamps = self.mc_time_stamps - self.mc_time_stamps[0]
        self.mc_end_cap_pos_obs = np.array(self.mc_end_cap_pos_obs) / 1000.  # mm to m
        self.T = len(self.mc_time_stamps)

        print(f"{mocap_statistics['all']}/{n_time_steps} has full motion capture data.")

    def correct_end_cap_positions_by_rod_length(self, end_cap_pos_obs, end_cap_pos_obs_mask):
        assert self.T == len(end_cap_pos_obs) == len(end_cap_pos_obs_mask)
        rod_length_w_end_caps = self.rod_length + self.mass_r*2
        corrected_end_cap_pos_obs = np.zeros_like(end_cap_pos_obs)
        corrected_end_cap_pos = np.zeros_like(end_cap_pos_obs)
        if end_cap_pos_obs_mask is None:
            end_cap_pos_obs_mask = np.ones((self.T, self.n_masses), dtype=bool)
        end_cap_pos_mask = np.copy(end_cap_pos_obs_mask)
        for t in range(self.T):
            position_mask = end_cap_pos_obs_mask[t]
            for end1, end2 in self.rod_end_ids:
                if position_mask[end1] and position_mask[end2]:
                    end1_pos = end_cap_pos_obs[t, end1]
                    end2_pos = end_cap_pos_obs[t, end2]
                    dist = np.linalg.norm(end2_pos - end1_pos, axis=-1, keepdims=True)
                    scale1 = rod_length_w_end_caps / dist
                    scale2 = self.rod_length / dist
                    center = (end1_pos + end2_pos) / 2.
                    new_end1_pos = center + (end1_pos - center) * scale1
                    new_end2_pos = center + (end2_pos - center) * scale1
                    corrected_end_cap_pos_obs[t, end1] = new_end1_pos
                    corrected_end_cap_pos_obs[t, end2] = new_end2_pos
                    corrected_end_cap_pos[t, end1] = center + (end1_pos - center) * scale2
                    corrected_end_cap_pos[t, end2] = center + (end2_pos - center) * scale2
                else:
                    end_cap_pos_mask[t][end1] = False
                    end_cap_pos_mask[t][end2] = False
        return corrected_end_cap_pos_obs, corrected_end_cap_pos, end_cap_pos_mask

    def transform_end_cap_positions_from_pose_estimation(self):
        # pose_folder = self.dataset_folder / 'smoothed_poses'
        pose_folder = self.dataset_folder / 'poses'
        n_end_caps = self.n_masses
        end_cap_positions = list()
        for i in range(n_end_caps):
            position_file = pose_folder / f'{i}_pos.npy'
            position = np.load(position_file)
            new_position = self.transform_positions(self.cfg['cam_extr'], position)
            end_cap_positions.append(new_position)
        end_cap_positions = np.stack(end_cap_positions, axis=1) * self.length_scale
        # correct end cap positions by rod length
        assert len(end_cap_positions) == self.T
        self.pe_end_cap_pos_obs_mask = np.ones((self.T, n_end_caps), dtype=bool)
        self.pe_end_cap_pos_obs_world, self.pe_end_cap_pos_world, self.pe_end_cap_pos_mask = self.correct_end_cap_positions_by_rod_length(
            end_cap_positions, self.pe_end_cap_pos_obs_mask)

    def center_end_cap_positions(self, end_cap_positions, end_cap_pos_mask):
        centered_end_cap_positions = list()
        for t in range(len(end_cap_positions)):
            end_cap_positions_at_t = end_cap_positions[t]
            if np.allclose(end_cap_pos_mask[t], True):
                center = np.mean(end_cap_positions_at_t, axis=-2, keepdims=True)
                center[..., 2] = 0
                centered_end_cap_positions.append(end_cap_positions_at_t - center)
            else:
                centered_end_cap_positions.append(end_cap_positions_at_t)
        return centered_end_cap_positions

    def compute_mc_extrinsic(self):
        mc2cam = self.cfg['mc2cam']
        self.cfg['mc_extr'] = self.cfg['cam_extr'] @ mc2cam

    def transform_end_cap_positions_from_motion_capture(self):
        # read mc end cap positions in mc frame
        mc_end_cap_pos_obs = self.mc_end_cap_pos_obs
        mc_end_cap_pos_obs_world = self.transform_positions(self.cfg['mc_extr'],
                                                            mc_end_cap_pos_obs) * self.length_scale
        self.mc_end_cap_pos_obs_world, self.mc_end_cap_pos_world, self.mc_end_cap_pos_mask = self.correct_end_cap_positions_by_rod_length(
            mc_end_cap_pos_obs_world, self.mc_end_cap_pos_obs_mask)

    def plot_controls(self, save_to=None):
        n_control = self.motor_cmd_speed.shape[1]

        for i in range(n_control):
            plt_x(self.motor_cmd_speed[:, i], title=f'Control_{i}', save_to=save_to)

    def compute_transform_to_set_foot_on_the_ground(self, end_cap_pos_world, end_cap_pos_mask):
        # get the lowest 2 point in each frame since at lease two feet are on the ground
        points = list()
        for t in range(self.T):
            if np.all(end_cap_pos_mask[t]):
                masses_pos = end_cap_pos_world[t]
                masses_pos_z = masses_pos[..., 2]
                feet = np.argsort(masses_pos_z)[:2]
                points.append(masses_pos[feet])
        points = np.array(points)

        pts = np.reshape(points, (-1, 3))
        num_pts = len(pts)
        early_stop_thresh = 0.95
        inlier_thresh = 0.002

        # run ransac to get the z axis
        origin = None
        plane_normal = None
        max_inlier_ratio = 0
        max_num_inliers = 0
        max_iterations = 1000

        for _ in range(max_iterations):
            # sample 3 points from the point cloud
            selected_indices = np.random.choice(num_pts, size=3, replace=False)
            selected_pts = pts[selected_indices]
            p1 = selected_pts[0]
            v1 = selected_pts[1] - p1
            v2 = selected_pts[2] - p1
            normal = np.cross(v1, v2)
            normal /= la.norm(normal)

            dist = np.abs((pts - p1) @ normal)
            num_inliers = np.sum(dist < inlier_thresh)
            inlier_ratio = num_inliers / num_pts

            if num_inliers > max_num_inliers:
                max_num_inliers = num_inliers
                origin = p1
                plane_normal = normal
                max_inlier_ratio = inlier_ratio

            if inlier_ratio > early_stop_thresh:
                break

        if plane_normal[2] < 0:
            plane_normal *= -1

        # randomly sample x_dir and y_dir given plane normal as z_dir
        x_dir = np.array([plane_normal[2], 0, -plane_normal[0]])
        x_dir /= la.norm(x_dir)
        y_dir = np.cross(plane_normal, x_dir)
        plane_frame = np.eye(4)
        plane_frame[:3, 0] = x_dir
        plane_frame[:3, 1] = y_dir
        plane_frame[:3, 2] = plane_normal

        # origin is the ground projection of center of mass of robot of first frame
        masses_pos = end_cap_pos_world[0]
        masses_pos_z = masses_pos[..., 2]
        feet = np.argsort(masses_pos_z)[:3]
        bottom_triangle_center = np.mean(masses_pos[feet], axis=0)
        bottom_triangle_center -= plane_normal * self.mass_r
        plane_frame[:3, 3] = bottom_triangle_center

        self.cfg['extr2floor'] = la.inv(plane_frame)
        cam_extr = np.copy(self.cfg['cam_extr'])
        cam_extr[0:3, 3] *= self.length_scale
        self.cfg['optim_cam_extr'] = self.cfg['extr2floor'] @ cam_extr

    @staticmethod
    def transform_positions(rot_mat, positions):
        T = 0
        n = 0
        if positions.ndim == 2:
            tmp_positions = positions
            ones = np.ones((len(positions), 1))
        else:
            T, n, dim = positions.shape
            tmp_positions = np.reshape(positions, (T * n, dim))
            ones = np.ones((T * n, 1))
        tmp_positions = np.concatenate((tmp_positions, ones), axis=1)
        new_positions = np.matmul(rot_mat, tmp_positions.transpose()).transpose()[..., 0:3]
        if positions.ndim == 3:
            new_positions = np.reshape(new_positions, (T, n, dim))
        return new_positions

    def center_mc_pe_end_cap_positions(self):
        self.mc_centerd_end_cap_pos_world = self.center_end_cap_positions(self.mc_end_cap_pos_world,
                                                                          self.mc_end_cap_pos_mask)
        self.pe_centered_end_cap_positions = self.center_end_cap_positions(self.pe_end_cap_pos_world,
                                                                           self.pe_end_cap_pos_mask)

    def set_feet_onto_the_ground(self):
        self.compute_transform_to_set_foot_on_the_ground(self.mc_end_cap_pos_world, self.mc_end_cap_pos_mask)
        self.mc_end_cap_pos_world = self.transform_positions(self.cfg['extr2floor'],
                                                             self.mc_end_cap_pos_world)
        self.mc_end_cap_pos_obs_world = self.transform_positions(self.cfg['extr2floor'],
                                                                 self.mc_end_cap_pos_obs_world)
        self.pe_end_cap_pos_world = self.transform_positions(self.cfg['extr2floor'],
                                                             self.pe_end_cap_pos_world)
        self.pe_end_cap_pos_obs_world = self.transform_positions(self.cfg['extr2floor'],
                                                                 self.pe_end_cap_pos_obs_world)
        # self.mc_end_cap_pos_world = self.correct_end_cap_positions_by_floor_z(self.mc_end_cap_pos_world,
        #                                                                       self.mc_end_cap_pos_mask)
        # self.pe_end_cap_pos_world = self.correct_end_cap_positions_by_floor_z(self.pe_end_cap_pos_world,
        #                                                                       self.pe_end_cap_pos_mask)

    def correct_end_cap_positions_by_floor_z(self, end_cap_pos_in_world, end_cap_pos_mask):
        floor_z = self.floor_z
        end_cap_r = self.mass_r
        corrected_end_cap_positions_list = list()
        for t in range(self.T):
            end_cap_positions = end_cap_pos_in_world[t]
            end_cap_mask = end_cap_pos_mask[t]
            corrected_end_cap_positions = np.copy(end_cap_positions)
            for rod_id in range(self.n_rods):
                end1_idx, end2_idx = self.rod_end_ids[rod_id]
                if end_cap_mask[end1_idx] and end_cap_mask[end2_idx]:
                    selected_end_cap_positions = end_cap_positions[[end1_idx, end2_idx]]
                    min_z = min(selected_end_cap_positions[:, 2])
                    delta_z = min_z - (floor_z + end_cap_r)
                    corrected_end_cap_positions[[end1_idx, end2_idx], 2] -= delta_z
            corrected_end_cap_positions_list.append(end_cap_positions)
        return np.array(corrected_end_cap_positions_list)


mc = MotionCapture(dataset_name='monday_roll15')
print(mc.cfg['extr2floor'])