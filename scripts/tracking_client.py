import os
import time
import copy
import importlib
import pprint
from argparse import ArgumentParser

import numpy as np
import numpy.linalg as la
import cv2
import networkx as nx
import open3d as o3d

import rospy
import rosgraph
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from tensegrity_perception.srv import init_tracker


def parse_argument():
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
    parser.add_argument("--num_dummy_points", type=int, default=50)
    parser.add_argument("--dummy_weights", type=float, default=0.1)
    parser.add_argument("--optimize_every_n_iters", type=int, default=1)
    parser.add_argument("--use_adaptive_weights", action="store_true")
    parser.add_argument("--filter_observed_pts", action="store_true")
    parser.add_argument("--add_physical_constraints", action="store_true")
    parser.add_argument("--add_ground_constraints", action="store_true")
    parser.add_argument("--save", action="store_true", help="save observation and qualitative results")
    parser.add_argument("-v", "--visualize", action="store_true")
    return parser


def main():
    if not rosgraph.is_master_online():
        print("roscore is not running! Run roscore first!")
        exit(0)

    rospy.init_node("tracking_client", anonymous=True)

    parser = parse_argument()
    args = parser.parse_args()

    video_path = os.path.join(args.dataset, args.video)
    prefixes = sorted([i.split('.')[0] for i in os.listdir(os.path.join(video_path, 'color'))])


if __name__ == "__main__":
    main()
