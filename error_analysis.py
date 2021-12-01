# read estimated pose
import os
import numpy as np
import quaternion

from utils.physics_utils import MassState, RodState
from utils.rotation_utils import np_rotmat_to_axis_angle, np_quat2mat, np_rotmat_of_two_v
from utils.visualize_utils import plt_multiple_x, plt_x


def down_sample(masses_pos, interval=40, scale=100.):
    gt_masses_pos = list()
    for t in range(0, len(masses_pos), interval):
        gt_masses_pos.append(masses_pos[t])
    gt_masses_pos = np.array(gt_masses_pos) / scale
    return gt_masses_pos


def filter(sensors, window_size=10):
    T = len(sensors)
    for t in range(T - window_size + 1):
        sensors[t] = np.average(sensors[t:t + window_size], axis=0)
    return sensors[:T - window_size + 1]

def plt_error_graph(rod_id, gt, est, window_size, label, unit='m/s'):
    T = min(len(gt), len(est))
    gt = gt[:T]
    est = est[:T]
    plt_multiple_x(gt, label_y=f'{label} ({unit})', title=f'rod {rod_id} {label} gt',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(est, label_y=f'{label} ({unit})', title=f'rod {rod_id} {label} est',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(gt - est, label_y=f'error ({unit})', title=f'rod {rod_id} {label} error',
                   label=['x', 'y', 'z'],
                   save_to=save_to)

    smoothed_est = filter(est, window_size=window_size)
    smoothed_gt = gt[: len(smoothed_est)]
    plt_multiple_x(smoothed_est, label_y=f'{label} ({unit})', title=f'rod {rod_id} {label} smoothed est',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(smoothed_gt - smoothed_est, label_y=f'error({unit})', title=f'rod {rod_id} {label} smoothed error',
                   label=['x', 'y', 'z'],
                   save_to=save_to)



# pose_type = 'smoothed_poses'
pose_type = 'poses'
pose_folder = f'/home/wang/github/nn_physics_simulation/tensegrity_perception/dataset/crawling_sim/{pose_type}'
# error_type = 'smoothed_errors'
error_type = 'errors'
save_to = f'/home/wang/github/nn_physics_simulation/tensegrity_perception/dataset/crawling_sim/{error_type}'

gt_pose_file = '/home/wang/github/nn_physics_simulation/data/3prism/go_right/test_experiments.pkl'
import pickle

with open(gt_pose_file, 'rb') as f:
    trajectory_list = pickle.load(f)
trajectory = trajectory_list[0]
masses_pos = MassState.get_position_from_warp_state(trajectory.mass_state_list)
gt_masses_pos = down_sample(masses_pos)
rods_pos = RodState.get_pos(trajectory.rod_state_list)
gt_rods_pos = down_sample(rods_pos)
gt_rods_lin_vel = down_sample(RodState.get_lin_vel(trajectory.rod_state_list))
gt_rods_ang_vel = down_sample(RodState.get_ang_vel(trajectory.rod_state_list))
gt_rods_quaternion = down_sample(RodState.get_quat(trajectory.rod_state_list), scale=1.)

# # 6 end caps
# for i in range(6):
#     pose_file = os.path.join(pose_folder, f'{i}_pos.npy')
#     estimated_cap_pos = np.load(pose_file)
#     T = len(estimated_cap_pos)
#     gt_cap_pos = gt_masses_pos[:T, i


#     error = gt_cap_pos - estimated_cap_pos
#     plt_multiple_x(error, label_y='error(m)', title=f'end cap {i} position error', label=['x', 'y', 'z'],
#                    save_to=save_to)
#     # print(estimated_cap_pos)

# 3 bars
delta_t = 0.04
window_size = 10
for rod_id, color in enumerate(['green', 'red', 'blue']):
    pose_file = os.path.join(pose_folder, f'{color}.npy')
    estimated_rod_pos = np.load(pose_file)
    rod_positions = estimated_rod_pos[:, 0:3, 3]
    rod_rot_mats = estimated_rod_pos[:, 0:3, 0:3]
    est_rod_lin_vel = (rod_positions[1:] - rod_positions[:-1]) / delta_t
    gt_rod_lin_vel = gt_rods_lin_vel[:, rod_id]
    plt_error_graph(rod_id, gt_rod_lin_vel, est_rod_lin_vel, window_size, label='lin vel')

    # dot_R = rod_rot_mats[:-1] - rod_rot_mats[1:]
    R = rod_rot_mats[:-1]
    # inv_R = np.linalg.inv(R)
    # ang_vel_mat = np.matmul(dot_R, inv_R) / delta_t

    # inv_R = np.linalg.inv(R)
    # delta_rot_mat = np.matmul(rod_rot_mats[1:], inv_R)
    rod_directions = -np.matmul(rod_rot_mats, [[0], [0], [1.]]).squeeze(-1)
    rod_ang_vel = list()
    rod_angles = list()
    T = len(estimated_rod_pos)
    for t in range(T - 1):
        m = np_rotmat_of_two_v(rod_directions[t], rod_directions[t + 1])
        axis, angle = np_rotmat_to_axis_angle(m)
        rod_angles.append(angle)
        # ||w|| * delta_t = theta
        # axis =  w / ||w||
        # thus, w = axis * ||w|| = axis * theta / delta_t

        ang_vel = axis * angle / delta_t
        rod_ang_vel.append(ang_vel)
    rod_ang_vel = np.asarray(rod_ang_vel)

    gt_lin_vel = gt_rods_lin_vel[:T - window_size, rod_id]
    rod_lin_vel = filter(est_rod_lin_vel, window_size=window_size)
    plt_multiple_x(gt_lin_vel - rod_lin_vel, label_y='error(m/s)', title=f'rod {rod_id} lin vel error',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(gt_lin_vel, label_y='velocity (m/s)', title=f'rod {rod_id} lin vel gt',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(rod_lin_vel, label_y='velocity (m/s)', title=f'rod {rod_id} lin vel pred',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_x(rod_angles, label='rot angle (rad)', title=f'rod {rod_id} rotation angle', save_to=save_to)

    gt_ang_vel = gt_rods_ang_vel[:T - window_size, rod_id]
    # rod_ang_vel = filter(rod_ang_vel)
    rod_ang_vel = rod_ang_vel[:T - window_size]
    plt_multiple_x(gt_ang_vel - rod_ang_vel, label_y='error (rad/s)', title=f'rod {rod_id} ang vel error',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(gt_ang_vel, label_y='velocity (rad/s)', title=f'rod {rod_id} ang vel gt',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(rod_ang_vel, label_y='velocity (rad/s)', title=f'rod {rod_id} ang vel pred',
                   label=['x', 'y', 'z'],
                   save_to=save_to)

    gt_pos = gt_rods_pos[:T - window_size + 1, rod_id]
    rod_positions = filter(rod_positions, window_size=window_size)
    plt_multiple_x(gt_pos - rod_positions, label_y='error(m)', title=f'rod {rod_id} pos error',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(gt_pos - gt_pos[0], label_y='position (m)', title=f'rod {rod_id} pos gt',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    plt_multiple_x(rod_positions - rod_positions[0], label_y='position (m)', title=f'rod {rod_id} pos est',
                   label=['x', 'y', 'z'],
                   save_to=save_to)
    # gt_quat = gt_rods_quaternion[:T, rod_id]
    # quats = quaternion.from_rotation_matrix(rod_rot_mats)
    # tmp_quats = list()
    # for q in quats:
    #     if q.w < 0:
    #         tmp_quats.append([-q.w, -q.x, -q.y, -q.z])
    #     else:
    #         tmp_quats.append([q.w, q.x, q.y, q.z])
    # tmp_quats = np.array(tmp_quats)
    # plt_x(np.linalg.norm(gt_quat - tmp_quats, axis=-1), label='error(norm)', title=f'rod {rod_id} quat error',
    #       save_to=save_to)
    gt_quat = gt_rods_quaternion[:T, rod_id]
    cos_sim_list = list()
    for t, q in enumerate(gt_quat):
        gt_rotmat = np_quat2mat(q)
        gt_axis = np.matmul(gt_rotmat, [[0], [0], [1]])
        pred_axis = np.matmul(rod_rot_mats[t], [[0], [0], [1]])

        cos_sim = np.vdot(gt_axis, pred_axis)
        cos_sim_list.append(cos_sim)
    plt_x(cos_sim_list, label='cosine', title=f'rod {rod_id} quat cos similarity',
          save_to=save_to)
