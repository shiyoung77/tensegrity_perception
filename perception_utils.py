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


def np_rotmat_of_two_v(v1, v2):
    v1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-20)
    v2 = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-20)

    if np.allclose(v1, -v2, atol=1e-3) or np.isclose(np.dot(v1, v2), -1, atol=1e-3):
        # rotmat = -np.eye(3)
        tmp_vs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        for tmp_v in tmp_vs:
            if np.isclose(np.dot(v1, tmp_v), -1, atol=1e-3) or np.isclose(np.dot(tmp_v, v2), -1, atol=1e-3):
                continue
            rotmat1 = np_rotmat_of_two_v(v1, tmp_v)
            rotmat2 = np_rotmat_of_two_v(tmp_v, v2)
            rotmat = np.matmul(rotmat2, rotmat1)
            break
    else:
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)  # sin is positive in [0, pi]
        c = v1.dot(v2)

        if np.isclose(s, 0) and c > 0:
            rotmat = np.eye(3)
        else:
            vHat = np_axis_to_skewsym(v)
            vHatSq = np.dot(vHat, vHat)
            rotmat = np.eye(3) + s * vHat + (1 - c) * vHatSq
    assert np.allclose(v2, rotmat.dot(v1))
    return rotmat


def np_axis_to_skewsym(v):
    '''
        Converts an axis into a skew symmetric matrix format.
    '''
    assert not np.isclose(np.linalg.norm(v), 0)
    v = v / np.linalg.norm(v)
    vHat = np.zeros((3, 3))
    vHat[0, 1], vHat[0, 2] = -v[2], v[1]
    vHat[1, 0], vHat[1, 2] = v[2], -v[0]
    vHat[2, 0], vHat[2, 1] = -v[1], v[0]
    return vHat


if __name__ == '__main__':
    pass
