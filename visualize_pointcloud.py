import os
import json
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

dataset = 'dataset/dummy'
with open(os.path.join(dataset, 'cam_config.json'), 'r') as f:
    cam_info = json.load(f)

prefix = 0
color_path = os.path.join(dataset, 'color', f'{prefix:04d}.png')
depth_path = os.path.join(dataset, 'depth', f'{prefix:04d}.png')

color_im = cv2.imread(color_path, cv2.IMREAD_COLOR)
# color_im = cv2.cvtColor(cv2.imread(color_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cam_info['depth_scale']

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].imshow(color_im)
ax[1].imshow(depth_im)
plt.show()

depth_im_o3d = o3d.geometry.Image(depth_im)
color_im_o3d = o3d.geometry.Image(color_im)

cam_intr_o3d = o3d.camera.PinholeCameraIntrinsic()
cam_intr_o3d.intrinsic_matrix = np.array(cam_info['cam_intr'])
rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im_o3d, depth_im_o3d, depth_scale=1,
                                                              convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, cam_intr_o3d)
o3d.visualization.draw_geometries([pcd])
