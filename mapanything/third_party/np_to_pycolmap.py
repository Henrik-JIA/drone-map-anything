# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified from https://github.com/facebookresearch/vggt

import numpy as np
import pycolmap

from mapanything.third_party.projection import project_3D_points_np


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    reproj_mask = None

    # 计算重投影误差
    if max_reproj_error is not None:
        # 把3D点投影回2D图像
        projected_points_2d, projected_points_cam = project_3D_points_np(
            points3d, extrinsics, intrinsics
        )
        # 计算投影位置和实际检测位置的差距
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        # 如果差距小于最大重投影误差，则设置为True
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    # 检查每帧是否有足够的有效点
    if masks.sum(1).min() < min_inlier_per_frame: # 默认每帧至少64个点
        print("Not enough inliers per frame, skip BA.")
        return None, None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    # 只添加在至少2张图像中可见的3D点
    inlier_num = masks.sum(0) # 统计每个点被多少张图看到，至少2张图可见的点才被添加
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0] # 获取至少2张图可见的点的索引

    # Only add 3D points that have sufficient 2D points
    for vidx in valid_idx:
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)

    num_points3D = len(valid_idx)
    camera = None
    # frame idx
    for fidx in range(N): # 遍历每张图像
        # set camera 
        # 如果shared_camera=True：所有图像共用1个相机对象；
        # 如果shared_camera=False：每张图像创建独立的相机对象。
        if camera is None or (not shared_camera):
            # 创建相机内参
            pycolmap_intri = _build_pycolmap_intri(
                fidx, intrinsics, camera_type, extra_params
            )

            # 创建相机对象
            camera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=pycolmap_intri,
                camera_id=fidx + 1,
            )

            # add camera 添加相机到重建对象
            reconstruction.add_camera(camera)

        # set image，w2c 变换，将世界坐标系下的外参转到相机坐标系下
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        # 创建图像对象，包含相机位姿和图像ID
        image = pycolmap.Image(
            id=fidx + 1,
            name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        # 为这张图像添加2D特征点，并关联到3D点
        points2D_list = [] # 创建一个空列表，用来存放这张照片中的所有2D特征点

        point2D_idx = 0 # 计数器，记录当前是这张照片中的第几个特征点

        # NOTE point3D_id start by 1
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1] # 获取这个3D点在照片中的索引

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all(): # 检查3D点是否合理，因为太远的点（比如坐标是10000）可能是异常值或错误
                if masks[fidx][original_track_idx]: # 检查这个点在当前照片中是否可见
                    # 记录"照片→3D点"的关系
                    # 第0个点(x,y) → 3D点id_1
                    # 第1个点(x,y) → 3D点id_2
                    # 第2个点(x,y) → 3D点id_3
                    # ...
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx] # 获取这个3D点在图像中的2D坐标
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id)) # 记录关联关系，将2D坐标和3D点ID关联起来

                    # 记录"3D点→照片"的关系
                    # 3D点id_1 → 第0个点(x,y)
                    # 3D点id_2 → 第1个点(x,y)
                    # 3D点id_3 → 第2个点(x,y)
                    # ...
                    # add element
                    track = reconstruction.points3D[point3D_id].track # 获取这个3D点的轨迹对象
                    track.add_element(fidx + 1, point2D_idx) # 这个3D点在第fidx + 1张照片中，是这张照片中的第point2D_idx个特征点
                    point2D_idx += 1

        # 断言检查，确保计数器和列表长度一致
        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list) 
            image.registered = True # 标记这张图像成功注册
        except:  # noqa
            print(f"frame {fidx + 1} is out of BA")
            image.registered = False # 标记这张图像注册失败

        # add image
        reconstruction.add_image(image)

    return reconstruction, valid_mask


def pycolmap_to_batch_np_matrix(
    reconstruction, device="cpu", camera_type="SIMPLE_PINHOLE"
):
    """
    Convert a PyCOLMAP Reconstruction Object to batched NumPy arrays.

    Args:
        reconstruction (pycolmap.Reconstruction): The reconstruction object from PyCOLMAP.
        device (str): Ignored in NumPy version (kept for API compatibility).
        camera_type (str): The type of camera model used (default: "SIMPLE_PINHOLE").

    Returns:
        tuple: A tuple containing points3D, extrinsics, intrinsics, and optionally extra_params.
    """

    num_images = len(reconstruction.images)
    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3))

    for point3D_id in reconstruction.points3D:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz

    extrinsics = []
    intrinsics = []

    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None

    for i in range(num_images):
        # Extract and append extrinsics
        pyimg = reconstruction.images[i + 1]
        pycam = reconstruction.cameras[pyimg.camera_id]
        matrix = pyimg.cam_from_world.matrix()
        extrinsics.append(matrix)

        # Extract and append intrinsics
        calibration_matrix = pycam.calibration_matrix()
        intrinsics.append(calibration_matrix)

        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    # Convert lists to NumPy arrays instead of torch tensors
    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)

    if camera_type == "SIMPLE_RADIAL":
        extra_params = np.stack(extra_params)
        extra_params = extra_params[:, None]

    return points3D, extrinsics, intrinsics, extra_params


########################################################


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=pycolmap_intri,
                camera_id=fidx + 1,
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=fidx + 1,
            name=f"image_{fidx + 1}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:  # noqa
            print(f"frame {fidx + 1} does not have any points")
            image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [
                intrinsics[fidx][0, 0],
                intrinsics[fidx][1, 1],
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
            ]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [
                focal,
                intrinsics[fidx][0, 2],
                intrinsics[fidx][1, 2],
                extra_params[fidx][0],
            ]
        )
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri
