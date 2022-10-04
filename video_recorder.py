# Author: Shiyang Lu, 2021
# Python >= 3.6 required, else modify os.makedirs and f-strings
# opencv-contrib-python: 4.2.0.34  (This version does not have Qthread issue.)
# Remove opencv-python before install opencv-contrib-python

import os
import json
from argparse import ArgumentParser
import pyrealsense2 as rs
import numpy as np
import cv2

import rospy
import rosgraph
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class RealSenseCamera:

    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = rs.config()

            # for L515
            # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
            # self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
            # self.config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)

            # for D435 and D415
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        self.bridge = CvBridge()

        # ros publisher
        self.color_im_pub = rospy.Publisher('rgb_images', Image, queue_size=10)
        self.depth_im_pub = rospy.Publisher('depth_images', Image, queue_size=10)

        self.record = False

    def run(self, rate, output_dir=None):
        if output_dir is not None:
            os.makedirs(os.path.join(output_dir, 'color'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)

        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)

        # depth align to color
        align = rs.align(rs.stream.color)
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = int(round(1 / depth_sensor.get_depth_scale()))

        print(color_intrinsics)
        self.cam_info = dict()
        self.cam_info['id'] = os.path.basename(output_dir)
        self.cam_info['im_w'] = color_intrinsics.width
        self.cam_info['im_h'] = color_intrinsics.height
        self.cam_info['depth_scale'] = depth_scale
        fx, fy = color_intrinsics.fx, color_intrinsics.fy
        cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
        self.cam_info['cam_intr'] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        cam_config_path = os.path.join(output_dir, 'config.json')
        with open(cam_config_path, 'w') as f:
            print(f"Camera info has been saved to: {cam_config_path}.")
            json.dump(self.cam_info, f, indent=4)

        count = 0
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        try:
            while not rospy.is_shutdown():
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                color_im = np.asanyarray(color_frame.get_data())  # rgb
                color_im_bgr = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
                depth_im = np.asanyarray(depth_frame.get_data())
                im_h, im_w = depth_im.shape

                # ROS publisher
                color_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR), 'rgb8')
                depth_msg = self.bridge.cv2_to_imgmsg(depth_im, 'mono16')
                self.color_im_pub.publish(color_msg)
                self.depth_im_pub.publish(depth_msg)

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                # depth_im_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_im, alpha=0.03), cv2.COLORMAP_JET)

                depth_im_vis = np.clip(depth_im.astype(np.float32) / depth_scale * 100, a_min=0, a_max=255)
                depth_im_vis = np.repeat(depth_im_vis, 3).reshape((im_h, im_w, 3)).astype(np.uint8)

                # Stack both images horizontally
                images = np.hstack((color_im_bgr, depth_im_vis))

                # Show images
                cv2.imshow('RealSense', images)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and output_dir is not None:
                    self.record = True
                    print("Start recording...")
                elif key == ord('s') and self.record:
                    self.record = False
                    print("Stop recording.")
                
                if self.record:
                    cv2.imwrite(os.path.join(output_dir, 'color', f"{count:04d}-color.png"), color_im_bgr)
                    cv2.imwrite(os.path.join(output_dir, 'depth', f"{count:04d}-depth.png"), depth_im)
                    count += 1
                    if count % 100 == 0:
                        print(f"{count} frames have been saved.")

                rate.sleep()
        finally:
            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    if not rosgraph.is_master_online():
        print("roscore is not running! Either comment out ROS related stuff or run roscore first!")
        exit(0)
    rospy.init_node("realsense")

    parser = ArgumentParser(description='SupervoxelContrast training')
    parser.add_argument('--datapath', default="/home/lsy/dataset/custom_ycb")
    parser.add_argument('-v', '--video', default="0008", help='video name')
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    rate = rospy.Rate(30)
    camera = RealSenseCamera()
    camera.run(rate, output_dir=os.path.join(args.datapath, args.video))
