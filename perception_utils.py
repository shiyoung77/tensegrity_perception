import time
import copy
import open3d as o3d
import numpy as np
import scipy.linalg as la

def timeit(f, n=1, need_compile=False):
    def wrapper(*args, **kwargs):
        if need_compile:  # ignore the first run if needs compile
            result = f(*args, **kwargs)
        print("------------------------------------------------------------------------")
        tic = time.time()
        for i in range(n):
            result = f(*args, **kwargs)
        total_time = time.time() - tic
        print(f"time elapsed: {total_time}s. Average running time: {total_time / n}s")
        return result
    return wrapper


def create_pcd(depth_im: np.ndarray,
               cam_intr: np.ndarray,
               color_im: np.ndarray = None,
               depth_scale: float = 1,
               depth_trunc: float = 1.5,
               cam_extr: np.ndarray = np.eye(4)):

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = cam_intr
    depth_im_o3d = o3d.geometry.Image(depth_im)
    if color_im is not None:
        color_im_o3d = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_im_o3d, depth_im_o3d,
            depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic=cam_extr)
    else:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_im_o3d, intrinsic_o3d, extrinsic=cam_extr,
                                                              depth_scale=depth_scale, depth_trunc=depth_trunc)
    return pcd


def icp(src, tgt, max_iter=30, init=np.eye(4), inverse=False):
    if len(src.points) > len(tgt.points):
        return icp(tgt, src, max_iter, init=la.inv(init), inverse=True)

    reg = o3d.pipelines.registration
    # reg = cph.registration
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    result_icp = reg.registration_icp(
        src, tgt, max_correspondence_distance=0.015,
        init=init,
        estimation_method=reg.TransformationEstimationPointToPoint(),
        criteria=reg.ICPConvergenceCriteria(max_iteration=max_iter))

    if inverse:
        result_icp.transformation = la.inv(result_icp.transformation)
    return result_icp


def plane_detection_ransac(pcd: o3d.geometry.PointCloud,
                           inlier_thresh: float,
                           max_iterations: int = 1000,
                           early_stop_thresh: float = 0.6,
                           visualize: bool = False):
    """detect plane (table) in a pointcloud for background removal

    Args:
        pcd (o3d.geometry.PointCloud): [input point cloud, assuming in the camera frame]
        inlier_thresh (float): [inlier distance threshold between a point to a plain]
        max_iterations (int): [max number of iteration to perform for RANSAC]
        early_stop_thresh (float): [inlier ratio to stop RANSAC early]

    Return:
        frame (np.ndarray): [z_dir is the estimated plane normal towards the camera, x_dir and y_dir randomly sampled]
        inlier ratio (float): [ratio of inliers in the estimated plane]
    """
    num_pts = len(pcd.points)
    pts = np.asarray(pcd.points)

    origin = None
    plane_normal = None
    max_inlier_ratio = 0
    max_num_inliers = 0

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

    # if plane_normal[2] < 0:
    #     plane_normal *= -1
    if plane_normal @ origin > 0:
        plane_normal *= -1

    # randomly sample x_dir and y_dir given plane normal as z_dir
    x_dir = np.array([-plane_normal[2], 0, plane_normal[0]])
    x_dir /= la.norm(x_dir)
    y_dir = np.cross(plane_normal, x_dir)
    plane_frame = np.eye(4)
    plane_frame[:3, 0] = x_dir
    plane_frame[:3, 1] = y_dir
    plane_frame[:3, 2] = plane_normal
    plane_frame[:3, 3] = origin

    if visualize:
        dist = (pts - origin) @ plane_normal
        pcd_vis = copy.deepcopy(pcd)
        colors = np.asarray(pcd_vis.colors)
        colors[np.abs(dist) < inlier_thresh] = np.array([0, 0, 1])
        colors[dist < -inlier_thresh] = np.array([1, 0, 0])

        plane_frame_vis = generate_coordinate_frame(plane_frame, scale=0.05)
        cam_frame_vis = generate_coordinate_frame(np.eye(4), scale=0.05)
        o3d.visualization.draw_geometries([cam_frame_vis, plane_frame_vis, pcd_vis])

    return plane_frame, max_inlier_ratio


def generate_coordinate_frame(T, scale=0.05):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(scale, center=np.array([0, 0, 0]))
    return mesh.transform(T)


def kabsch(Q: np.ndarray, P: np.ndarray):
    """
    compute rigid transformation T = [R, t], s.t. Q = R @ P + t
    Q (np.ndarray): shape=(3, N)
    P (np.ndarray): shape=(3, N)
    """
    P_mean = P.mean(1, keepdims=True)  # (3, 1)
    Q_mean = Q.mean(1, keepdims=True)  # (3, 1)

    H = (Q - Q_mean) @ (P - P_mean).T
    U, D, V_t = la.svd(H)
    R = U @ V_t

    # ensure that R is in the right-hand coordinate system, very important!!!
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    d = np.sign(la.det(U @ V_t))
    R = U @ np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, d]
    ], dtype=np.float64) @ V_t
    t = Q_mean - R @ P_mean

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T
