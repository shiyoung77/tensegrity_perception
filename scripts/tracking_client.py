#!/home/lsy/anaconda3/envs/tensegrity/bin/python

import os
import json
import time
import copy
from argparse import ArgumentParser

import numpy as np
import cv2
import open3d as o3d

import rospy
import rosgraph
import cv_bridge
from std_msgs.msg import Float64MultiArray
from tensegrity_perception.srv import InitTracker, InitTrackerRequest, InitTrackerResponse

from perception_utils import create_pcd


def init_tracker(rgb_im: np.ndarray, depth_im: np.ndarray, cable_lengths: np.ndarray):
    bridge = cv_bridge.CvBridge()
    rgb_msg = bridge.cv2_to_imgmsg(rgb_im, encoding="rgb8")
    depth_msg = bridge.cv2_to_imgmsg(depth_im, encoding="mono16")
    cable_length_msg = Float64MultiArray()
    cable_length_msg.data = cable_lengths.tolist()

    request = InitTrackerRequest()
    request.rgb_im = rgb_msg
    request.depth_im = depth_msg
    request.cable_lengths = cable_length_msg

    service_name = "init_tracker"
    rospy.loginfo(f"Waiting for {service_name} service...")
    rospy.wait_for_service(service_name)
    rospy.loginfo(f"Found {service_name} service.")
    try:
        init_tracker_srv = rospy.ServiceProxy(service_name, InitTracker)
        rospy.loginfo("Request sent. Waiting for response...")
        response: InitTrackerResponse = init_tracker_srv(request)
        return response.initialized
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")


def main():
    if not rosgraph.is_master_online():
        print("roscore is not running! Run roscore first!")
        exit(0)

    rospy.init_node("tracking_client", anonymous=True)

    dataset = '/home/lsy/dataset/tensegrity'
    video = 'pebbles9'
    video_path = os.path.join(dataset, video)

    color_path = os.path.join(video_path, 'color', '0000.png')
    depth_path = os.path.join(video_path, 'depth', '0000.png')
    info_path = os.path.join(video_path, 'data', '0000.json')

    rgb_im = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 4000

    cam_intr = np.array([
        [601.67626953125, 0.0, 325.53509521484375],
        [0.0, 601.7260131835938, 248.60618591308594],
        [0.0, 0.0, 1.0]
    ])

    # pcd = create_pcd(depth_im, cam_intr, rgb_im)
    # o3d.visualization.draw_geometries_with_editing([pcd])

    print(depth_im[100, 100])
    with open(info_path, 'r') as f:
        info = json.load(f)
    cable_lengths = np.zeros(len(info['sensors']))
    for key, sensor_data in info['sensors'].items():
        cable_lengths[int(key)] = sensor_data['length']

    init_tracker(rgb_im, depth_im, cable_lengths)


if __name__ == "__main__":
    main()
