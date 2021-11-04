# read estimated pose
import os
import numpy as np

from utils.physics_utils import MassState
from utils.visualize_utils import plt_multiple_x

pose_folder = '/home/wang/github/nn_physics_simulation/tensegrity_perception/dataset/crawling_sim/poses'

gt_pose_file = '/home/wang/github/nn_physics_simulation/data/3prism/go_right/test_experiments.pkl'
import pickle
with open(gt_pose_file, 'rb') as f:
    trajectory_list = pickle.load(f)
trajectory = trajectory_list[0]
masses_pos = MassState.get_position_from_warp_state(trajectory.mass_state_list)
gt_masses_pos = list()
for t in range(0, len(masses_pos), 40):
    gt_masses_pos.append(masses_pos[t])
gt_masses_pos = np.array(gt_masses_pos) / 100.

save_to = '/home/wang/github/nn_physics_simulation/tensegrity_perception/dataset/crawling_sim/errors/'
# 6 end caps
for i in range(6):
    pose_file = os.path.join(pose_folder, f'{i}_pos.npy')
    estimated_cap_pos = np.load(pose_file)
    T = len(estimated_cap_pos)
    gt_cap_pos = gt_masses_pos[:T, i]
    error = gt_cap_pos - estimated_cap_pos
    plt_multiple_x(error, label_y='error(m)', title=f'end cap {i} position error', label=['x', 'y', 'z'], save_to=save_to)
    # print(estimated_cap_pos)


