import argparse
from os import path as osp
import sys
import mmcv
import numpy as np
import os
from collections import OrderedDict, deque
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
# from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import Dict, List, Optional, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.map_expansion.bitmap import BitMap
from matplotlib.patches import Polygon as mPolygon

from shapely import affinity, ops
# from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box, MultiLineString
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import networkx as nx

import open3d as o3d
import pickle

import cv2

from mmdet3d.core.points import BasePoints, get_points_type

sys.path.append('.')


from mmdet3d.datasets.pipelines import LoadPointsFromMultiSweeps, LoadPointsFromFile

import copy



class CNuScenesMapExplorer(NuScenesMapExplorer):
    def __ini__(self, *args, **kwargs):
        super(self, CNuScenesMapExplorer).__init__(*args, **kwargs)

    def _get_centerline(self,
                           patch_box: Tuple[float, float, float, float],
                           patch_angle: float,
                           layer_name: str,
                           return_token: bool = False) -> dict:
        """
         Retrieve the centerline of a particular layer within the specified patch.
         :param patch_box: Patch box defined as [x_center, y_center, height, width].
         :param patch_angle: Patch orientation in degrees.
         :param layer_name: name of map layer to be extracted.
         :return: dict(token:record_dict, token:record_dict,...)
         """
        if layer_name not in ['lane','lane_connector']:
            raise ValueError('{} is not a centerline layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_api, layer_name)

        centerline_dict = dict()
        for record in records:
            if record['polygon_token'] is None:
                # import ipdb
                # ipdb.set_trace()
                continue
            polygon = self.map_api.extract_polygon(record['polygon_token'])

            # if polygon.intersects(patch) or polygon.within(patch):
            #     if not polygon.is_valid:
            #         print('within: {}, intersect: {}'.format(polygon.within(patch), polygon.intersects(patch)))
            #         print('polygon token {} is_valid: {}'.format(record['polygon_token'], polygon.is_valid))

            # polygon = polygon.buffer(0)

            if polygon.is_valid:
                # if within or intersect :

                new_polygon = polygon.intersection(patch)
                # new_polygon = polygon

                if not new_polygon.is_empty:
                    centerline = self.map_api.discretize_lanes(
                            record, 0.5)
                    centerline = list(self.map_api.discretize_lanes([record['token']], 0.5).values())[0]
                    centerline = LineString(np.array(centerline)[:,:2].round(3))
                    if centerline.is_empty:
                        continue
                    centerline = centerline.intersection(patch)
                    if not centerline.is_empty:
                        centerline = \
                            to_patch_coord(centerline, patch_angle, patch_x, patch_y)
                        
                        # centerline.coords = np.array(centerline.coords).round(3)
                        # if centerline.geom_type != 'LineString':
                            # import ipdb;ipdb.set_trace()
                        record_dict = dict(
                            centerline=centerline,
                            token=record['token'],
                            incoming_tokens=self.map_api.get_incoming_lane_ids(record['token']),
                            outgoing_tokens=self.map_api.get_outgoing_lane_ids(record['token']),
                        )
                        centerline_dict.update({record['token']: record_dict})
        return centerline_dict

def to_patch_coord(new_polygon, patch_angle, patch_x, patch_y):
    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                  origin=(patch_x, patch_y), use_radians=False)
    new_polygon = affinity.affine_transform(new_polygon,
                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    return new_polygon



def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)

def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose

def get_pose(rotation, translation, inv=False, flat=False):
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)

    return get_transformation_matrix(R, t, inv=inv)

def parse_pose(record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

def kalman_pnts_to_grid(pnts_tr_all, s=1):
    X_MIN, X_MAX = -15, 15
    Y_MIN, Y_MAX = -30, 30
    nx = 100*s
    ny = 200*s
    RES =  (X_MAX - X_MIN)/nx
    # --- constants -------------------------------------------------------------
    # SIGMA_START   = 0.05                      # σ0  (prior variance)
    # SIGMA_MEASURE = 0.5                       # σz² (measurement variance)
    # MU_START      = 0.1                       # μ0  (prior mean)
    # INV_SIG0      = 1.0 / SIGMA_START
    # INV_SIGZ      = 1.0 / SIGMA_MEASURE
    SIGMA_MEASURE =  0.5                     # σz² (measurement variance)
    SIGMA_START   = 10*SIGMA_MEASURE                      # σ0  (prior variance)

    # SIGMA_START   = 10                      # σ0  (prior variance)
    # SIGMA_MEASURE = 2                     # σz² (measurement variance)
    MU_START      = 0.2                       # μ0  (prior mean)
    INV_SIG0      = 1.0 / SIGMA_START**2
    INV_SIGZ      = 1.0 / SIGMA_MEASURE**2

    # --- prepare points --------------------------------------------------------
    alpha = np.clip(1 / np.sin(np.abs(pnts_tr_all[:, 4])) / 5, 0, 1)   # shape (N,)
    # alpha = 1
    x, y, z = pnts_tr_all[:, 0], pnts_tr_all[:, 1], pnts_tr_all[:, 3] * alpha

    ix = ((x - X_MIN) / RES).round().astype(int)
    iy = ((y - Y_MIN) / RES).round().astype(int)

    # --- keep only points that fall inside the grid ---------------------------
    inside = (0 <= ix) & (ix < nx) & (0 <= iy) & (iy < ny)
    ix, iy, z = ix[inside], iy[inside], z[inside]

    # --- collapse 2-D indices to 1-D so we can use bincount -------------------
    flat = iy * nx + ix                            # shape (N,)
    n_cells = nx * ny

    # accumulate   n_j   and   Σz_j   per cell
    counts   = np.bincount(flat, minlength=n_cells)                    # n_j
    sum_z    = np.bincount(flat, weights=z, minlength=n_cells)         # Σ z_i  per cell

    # --- closed-form posterior -------------------------------------------------
    inv_sigma_post = INV_SIG0 + counts * INV_SIGZ                      # 1/σ²_post
    sigma_post     = np.where(counts, 1.0 / inv_sigma_post, np.nan)    # σ²_post  or NaN
    mu_post        = np.where(
        counts,
        (MU_START * INV_SIG0 + sum_z * INV_SIGZ) * sigma_post,         # μ_post
        np.nan)

    # --- reshape back to (ny, nx) ---------------------------------------------
    grid_        = mu_post.reshape(ny, nx).astype(np.float32)
    sigma_grid_  = sigma_post.reshape(ny, nx).astype(np.float32)
    return grid_


def idw_grid_interp(grid, k=8, d=10):
    from scipy.spatial import cKDTree

    # coordinates of valid pixels
    pts       = np.column_stack(np.nonzero(~np.isnan(grid)))
    vals      = grid[~np.isnan(grid)]
    tree      = cKDTree(pts)

    qry_y, qry_x = np.nonzero(np.isnan(grid))
    d, idx = tree.query(np.column_stack((qry_y, qry_x)),
                        k=k, distance_upper_bound=d)

    sentinel = len(vals)              # 9078 in your run
    mask     = (idx == sentinel)     # True where neighbour is missing

    # replace invalid indices with 0 so we can index vals
    idx[mask] = 0
    w         = 1/(d + 1e-6 )
    w[mask]   = 0                     # those neighbours contribute 0 weight

    num = np.sum(w * vals[idx], axis=1)
    den = np.sum(w, axis=1)
    grid_filled = grid.copy()
    grid_filled[qry_y, qry_x] = np.divide(num, den, out=np.full_like(num, np.nan), where=den>0)

    return grid_filled


def get_next_key(odict, current_key):
    keys = list(odict.keys())
    try:
        idx = keys.index(current_key)
        return keys[idx + 1]
    except (ValueError, IndexError):
        return None  


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         nusc_maps, 
                         map_explorer,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         point_cloud_range=[-15.0, -30.0,-10.0, 15.0, 30.0, 10.0]):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    kkkk = 0

    ### distant sweeps preloaded collection
    # distant_sweeps_collection = OrderedDict()
    # distant_sweeps_collection_idx = []

    central_idx = ""
    next_idx = []
    prev_idx = []

    sweeps_db = dict()

    """
    [ x  |  |  | ]
    [ |  x  |  |  | ]
    [ |  |  x  |  |  | ]
    [ |  |  |  x  |  |  | ]
       [ |  |  |  x  |  |  | ]
          [ |  |  |  x  |  |  | ]
             [ |  |  |  x  |  |  | ]
                [ |  |  |  x  |  | ]
                   [ |  |  |  x  | ]
                      [ |  |  |  x ]
    """
    to_init = False

    for sample in mmcv.track_iter_progress(nusc.sample):
        # print(sample)
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        kkkk+=1
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        egolidar = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        egolidarflat_from_world = parse_pose(egolidar, flat=True, inv=True)
        
        mmcv.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        ##
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'map_location': map_location,
            'scene_token': sample['scene_token'],  # temporal related info
            'pose_inverse': egolidarflat_from_world.tolist(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1
        to_init = False
        if frame_idx == 1:
            to_init = True 


        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps


        ##### begin lidar features generation

        D_th = 0.65

        dummy_object = LoadPointsFromMultiSweeps(sweeps_num=10, load_dim=5)

        
        def load_sweep(idd, sd_rec):
            min_bound = np.array([-70, -70, -3]).astype(np.float32)
            max_bound = np.array([70, 70, 3]).astype(np.float32)

            # t_load_st = time.time()
            data_path = str(nusc.get_sample_data_path(sd_rec['token']))
            points_sweep = dummy_object._load_points(data_path)
            
            # print((time.time() - t_load_st)*1000, "\n\n")       
            points_sweep = np.copy(points_sweep).reshape(-1, dummy_object.load_dim)
            # if self.remove_close:
            points_sweep = dummy_object._remove_close(points_sweep)
            # sweep_points_list.append(points_sweep)
            pcd = o3d.t.geometry.PointCloud()
            pcd.point['positions'] = o3d.core.Tensor(points_sweep[:, :3], dtype=o3d.core.Dtype.Float32)
            pcd.point['intensities'] = o3d.core.Tensor(points_sweep[:, [3]], dtype=o3d.core.Dtype.Float32)

            mask = (pcd.point["positions"] >= o3d.core.Tensor(min_bound)).all(1) & \
            (pcd.point["positions"] <= o3d.core.Tensor(max_bound)).all(1)
            pcd_in_box = pcd.select_by_mask(mask)
            
            pcd = pcd_in_box.voxel_down_sample(0.2)

            pnts = pcd.point["positions"].numpy()
            ints = pcd.point["intensities"].numpy()
            return {
                "pnts" : pnts,
                "ints" : ints,
                "sd_rec" : sd_rec,
                "id" : idd
            }
        
        def search_sweeps(start_idx, field, list_to_update, N, M=500):
            sd_rec_init = nusc.get('sample_data', start_idx)
            
            sd_rec = copy.deepcopy(sd_rec_init)
            pose_record_prev = copy.deepcopy(nusc.get('ego_pose', sd_rec['ego_pose_token']))
            j = 0
            while len(list_to_update) < N and j < M:
                if not sd_rec[field] == '':
                    # sweep = obtain_sensor2top(nusc, sd_rec[field], l2e_t,
                    #                         l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                    D = np.linalg.norm(np.array(pose_record["translation"]) - np.array(pose_record_prev["translation"]))
                    # obtain annotation
                    # import ipdb;ipdb.set_trace()
                    iddd =  sd_rec[field]
                    sd_rec =  copy.deepcopy(nusc.get('sample_data', sd_rec[field]))
                    if D > D_th:
                        list_to_update.append(iddd)
                        pose_record_prev =  copy.deepcopy(pose_record)
                    
                    j+=1
                    
                else:
                    break
            # else:
                
        
        def update_db(all_idx, sweeps_db):
            new = set(all_idx) - set(sweeps_db.keys())
            to_del = set(sweeps_db.keys()) - set(all_idx)
            for i in to_del:
                sweeps_db.pop(i)
            
            for n in new:
                sd_rec = nusc.get('sample_data', n)
                # sweep = obtain_sensor2top(nusc, n, l2e_t,
                #                             l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps_db[n] = load_sweep(n, sd_rec)


                    

        N = 25
        current_idx = sample['data']['LIDAR_TOP']
        sd_rec_current = nusc.get('sample_data', current_idx)
        print(sd_rec_current["prev"], current_idx, sd_rec_current["next"])

        
        if len(next_idx) > 0:
            # next_t = np.array(nusc.get('ego_pose', sweeps_db[next_idx[0]]["sd_rec"]["ego_pose_token"])["translation"])
            # current_t = np.array(nusc.get('ego_pose',sd_rec_current["ego_pose_token"])["translation"])
            # central_t = np.array(nusc.get('ego_pose', sweeps_db[central_idx]["sd_rec"]["ego_pose_token"])["translation"])
            found = False
            for i in range(len(next_idx)):
                next_t = int(sweeps_db[next_idx[i]]["sd_rec"]["timestamp"])
                current_t = int(sd_rec_current["timestamp"])
                central_t = int(sweeps_db[next_idx[i-1]]["sd_rec"]["timestamp"])
                # print(next_t, current_t, central_t)
                
                if  (central_t <= current_t ) and (current_t<= next_t) :

                    prev_idx.extend(next_idx[:i])
                    prev_idx = prev_idx[-N:]
                    # central_idx = next_idx[i]
                    del next_idx[:i]
                    print("search starts at len:", len(next_idx), next_idx)

                    search_sweeps(next_idx[-1], "next", next_idx, N)
                    found = True
                    break
            if not found:
                print("PUPUPUPU!\nPUPUPUPU\n!!!")
                next_idx = [current_idx]
                prev_idx = []
                search_sweeps(current_idx, "next", next_idx, N)
                search_sweeps(current_idx, "prev", prev_idx, N)
                
        if to_init:
            print("init")
            # central_idx = current_idx
            next_idx = [current_idx]
            prev_idx = []
            sweeps_db = dict()
            
            search_sweeps(next_idx[0], "next", next_idx, N)
            # search_sweeps(central_idx, "prev", prev_idx, N)
            

        
                # nex
        # central_idx = current_idx
        # next_idx = []
        # prev_idx = []
        
        # search_sweeps(central_idx, "next", next_idx, N)
        # search_sweeps(central_idx, "prev", prev_idx, N)
            
        
        # sync db
        # sweeps_db = dict()
        all_idx = [*prev_idx[::], *next_idx ]
        print("all_idx", len(all_idx), len(prev_idx), len(next_idx))
        update_db(all_idx, sweeps_db)
         
        dd = dict()
        if not current_idx in sweeps_db.keys():
            dd = {
                current_idx : load_sweep(current_idx, sd_rec_current)
            }
            

        ts = info['timestamp'] 


        pnts_tr_all = []

        for idx, d in [*sweeps_db.items(), *dd.items()]:
            # d = sweeps_db[idx]
            pnts,ints = d["pnts"],d["ints"]
            sweep = obtain_sensor2top(nusc, idx, l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
            
            R = sweep["sensor2lidar_rotation"]
            t = sweep["sensor2lidar_translation"]

            R_es = Quaternion(sweep['sensor2ego_rotation']).rotation_matrix
            t_es = sweep['sensor2ego_translation']
            pnts_e = pnts @ R_es.T + t_es
            mask = ~(((pnts_e[:, 0] < 3)&(pnts_e[:, 0] > -1)&(np.abs(pnts_e[:, 1]) < 1.5)) | (pnts_e[:, 2] > 0.4) | (pnts_e[:, 2] < -1))
            pnts = pnts[mask]
            ints = ints[mask]

            pnts_tr = pnts @ R.T + t
            data = np.zeros(shape=(pnts_tr.shape[0], 6))
            data[:, :3] = pnts_tr[:, :3]
            data[:, 3] = np.clip(ints[:, 0]/40, 0, 10)
            data[:, 4] = np.arctan2(pnts[:, 2], np.linalg.norm(pnts[:, :2], axis=1))
            data[:, 5] = np.linalg.norm(pnts[:, :3], axis=1)
            pnts_tr_all.append(data)

        pnts_tr_all = np.vstack(pnts_tr_all)

        grid = kalman_pnts_to_grid(pnts_tr_all, s=2)
        grid = idw_grid_interp(grid, k=12, d=20)

        log_grid = np.log(grid + 0.001)
        # blur = cv2.GaussianBlur(log_grid, ksize=(7, 7), sigmaX=0.2)

        blur = cv2.medianBlur(log_grid, 3)
        
        # 3. Sobel gradients ----------------------------------------------------------
        # x- and y-direction, 16-bit signed output to avoid clipping
        sobel_ksize=7
        grad_x = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=0, ksize=sobel_ksize,  scale=1.0/pow(2, sobel_ksize*2 - 1 - 2))
        grad_y = cv2.Sobel(blur, cv2.CV_64F, dx=0, dy=1, ksize=sobel_ksize,  scale=1.0/pow(2, sobel_ksize*2 - 1 - 2))
        grad_x[np.isnan(grad_x)] = 0
        grad_y[np.isnan(grad_y)] = 0
        grad_x =cv2.resize(grad_x, (100, 200))*2
        grad_y =cv2.resize(grad_y, (100, 200))*2
        grad_x = np.tanh(grad_x/1.2)*1.2
        grad_y = np.tanh(grad_y/1.2)*1.2
        # nusc.pa

        lidar_bev_maps = np.dstack((grad_x, grad_y))
        lidar_bev_map_path = f"{nusc.dataroot}/lidar_bev_maps/{sample['token']}.npy"
        if not os.path.isfile(lidar_bev_map_path):
            np.save(lidar_bev_map_path,  lidar_bev_maps)
        info["lidar_bev_maps_path"] = lidar_bev_map_path
        # info["segmap_path"] = f"/home/vasily/nuScenes/segmaps/{kkkk}.npy"

        # import matplotlib.pyplot as plt
        # # plt.imshow(lidar_bev_maps[:, :, 0])
        # plt.imshow(lidar_bev_maps[:, :, 1])
        # plt.show()

        ##### end lidar features generation



        # obtain annotation
        # import ipdb;ipdb.set_trace()
        info = obtain_vectormap(nusc_maps, map_explorer, info, point_cloud_range)
        
        info = get_static_layers(nusc, nusc_maps, info, 
                          map_location
                          )
        
        # if len(sweeps) > 0:
        #     np.save(f"/home/vasily/nuScenes_mini/lidar_debug/{kkkk}/segmap.npy", info["segmap"])

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
        sh = h / h_meters
        sw = w / w_meters

        return np.float32([
            [ sw,  0.,          w/2.],
            [ 0.,  sw, h*offset+h/2.],
            [ 0.,  0.,            1.]
        ])

def get_static_layers(nusc, nusc_map, 
                      sample,
                      location, 
                      layers=['ped_crossing', 'drivable_area'], 
                      point_cloud_range = [-10.0, -10.0,-10.0, 10.0, 10.0, 10.0],
                      bev={'h': 200, 'w': 100, 'h_meters': 60, 'w_meters': 30, 'offset': 0.0},
                      canvas_size = (200, 100)):
    import cv2
    INTERPOLATION = cv2.LINE_8
    
    h, w = canvas_size
    V = get_view_matrix(**bev)
    
    S = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    # print(sample)
    
    
    lidar2ego = np.eye(4)
    lidar2ego[:3,:3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = sample['lidar2ego_translation']
    ego2global = np.eye(4)
    ego2global[:3,:3] = Quaternion(sample['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = sample['ego2global_translation']
    lidar2global = ego2global @ lidar2ego
    lidar2global_translation = list(lidar2global[:3,3])
    T_lg = np.linalg.inv(lidar2global)

    map_pose = lidar2global_translation[:2]
    
    patch_w = point_cloud_range[3] - point_cloud_range[0]
    patch_h = point_cloud_range[4] - point_cloud_range[1]
    # patch_size = (patch_h, patch_w)
    # patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    records_in_patch = nusc_map[location].get_records_in_radius(map_pose[0], map_pose[1], max(patch_h, patch_w)*2, layers, 'intersect')

    result = list()

    M = V @ S @ T_lg
    for layer in layers:
        render = np.zeros((h, w), dtype=np.uint8)

        for r in records_in_patch[layer]:
            polygon_token = nusc_map[location].get(layer, r)
            
            # if layer

            # if layer == 'drivable_area': polygon_tokens = polygon_token['polygon_tokens']
            # else: polygon_tokens = [polygon_token['polygon_token']]

            # for p in polygon_tokens:
            if "polygon_token" in polygon_token.keys():
                polygon = nusc_map[location].extract_polygon(polygon_token["polygon_token"])
                multipolygon = MultiPolygon([polygon])
            else:
                multipolygon =  MultiPolygon([nusc_map[location].extract_polygon(p) for p in polygon_token["polygon_tokens"]])
            for polygon in multipolygon.geoms:
                exterior = np.array(polygon.exterior.coords)
                exterior =  exterior @ M[:2, :2].T + M[:2, 3]
                # cv2.polylines(image, [exterior], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.fillPoly(render, [exterior.astype(np.int32)], color=1)  # Optional: Fill the polygon

                # Optionally, draw holes if there are interiors
                for interior in polygon.interiors:
                    hole = np.array(interior.coords)
                    hole = hole @ M[:2, :2].T + M[:2, 3]
                    # cv2.polylines(render, [hole], isClosed=True, color=0, thickness=2)
                    cv2.fillPoly(render, [hole.astype(np.int32)], color=0)  # Fill holes with background


            # exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]
            # exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in exteriors]
            # exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in exteriors]
            # exteriors = [V @ S @ T_lg @ p for p in exteriors]
            # exteriors = [p[:2].round().astype(np.int32).T for p in exteriors]

            # cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

            # interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
            # interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in interiors]
            # interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in interiors]
            # interiors = [V @ S @ lidar2global @ p for p in interiors]
            # interiors = [p[:2].round().astype(np.int32).T for p in interiors]

            # cv2.fillPoly(render, interiors, 0, INTERPOLATION)

        result.append(render)

    path = f"{nusc.dataroot}/segmaps/{sample['token']}.npy"
    if not os.path.isfile(path):
        np.save(path,  255 * np.stack(result, -1))
    sample["segmap_path"] = path
    return sample

def obtain_vectormap(nusc_maps, map_explorer, info, point_cloud_range):
    # import ipdb;ipdb.set_trace()
    lidar2ego = np.eye(4)
    lidar2ego[:3,:3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = info['lidar2ego_translation']
    ego2global = np.eye(4)
    ego2global[:3,:3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = info['ego2global_translation']

    lidar2global = ego2global @ lidar2ego

    lidar2global_translation = list(lidar2global[:3,3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

    location = info['map_location']
    ego2global_translation = info['ego2global_translation']
    ego2global_rotation = info['ego2global_rotation']

    patch_h = point_cloud_range[4]-point_cloud_range[1]
    patch_w = point_cloud_range[3]-point_cloud_range[0]
    patch_size = (patch_h, patch_w)
    vector_map = VectorizedLocalMap(nusc_maps[location], map_explorer[location],patch_size)
    map_anns = vector_map.gen_vectorized_samples(lidar2global_translation, lidar2global_rotation)
    # import ipdb;ipdb.set_trace()
    info["annotation"] = map_anns
    return info


class VectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }
    def __init__(self,
                 nusc_map,
                 map_explorer,
                 patch_size,
                 map_classes=['divider','ped_crossing','boundary','centerline'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 centerline_classes=['lane_connector','lane'],
                 use_simplify=True,
                 ):
        super().__init__()
        self.nusc_map = nusc_map
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.centerline_classes = centerline_classes
        self.patch_size = patch_size


    def gen_vectorized_samples(self, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        '''
        
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        # import ipdb;ipdb.set_trace()
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        map_dict = {'divider':[],'ped_crossing':[],'boundary':[],'centerline':[]}
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes)
                line_instances_dict = self.line_geoms_to_instances(line_geom)     
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        map_dict[vec_class].append(np.array(instance.coords))
                        # vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    # vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    # import ipdb;ipdb.set_trace()
                    map_dict[vec_class].append(np.array(instance.coords))
                    # vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            elif vec_class =='centerline':
                centerline_geom = self.get_centerline_geom(patch_box, patch_angle, self.centerline_classes)
                centerline_list = self.centerline_geoms_to_instances(centerline_geom)
                for instance in centerline_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        # import ipdb;ipdb.set_trace()
        return map_dict
    def get_centerline_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.centerline_classes:
                return_token = False
                layer_centerline_dict = self.map_explorer._get_centerline(
                patch_box, patch_angle, layer_name, return_token=return_token)
                if len(layer_centerline_dict.keys()) == 0:
                    continue
                # import ipdb;ipdb.set_trace()
                map_geom.update(layer_centerline_dict)
        return map_geom
    def get_map_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle)
                # map_geom.append((layer_name, geoms))
                map_geom[layer_name] = geoms
        return map_geom

    def get_divider_line(self,patch_box,patch_angle,layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer.map_api, layer_name)
        for record in records:
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_contour_line(self,patch_box,patch_angle,layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer.map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list


    def get_ped_crossing_line(self, patch_box, patch_angle):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer.map_api, 'ped_crossing')
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom.items():
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def ped_poly_geoms_to_instances(self, ped_geom):
        # ped = ped_geom[0][1]
        # import ipdb;ipdb.set_trace()
        ped = ped_geom['ped_crossing']
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)


    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom['road_segment']
        lanes = polygon_geom['lane']
        # import ipdb;ipdb.set_trace()
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def centerline_geoms_to_instances(self,geoms_dict):
        centerline_geoms_list,pts_G = self.union_centerline(geoms_dict)
        # vectors_dict = self.centerline_geoms2vec(centerline_geoms_list)
        # import ipdb;ipdb.set_trace()
        return self._one_type_line_geom_to_instances(centerline_geoms_list)


    def centerline_geoms2vec(self, centerline_geoms_list):
        vector_dict = {}
        # import ipdb;ipdb.set_trace()
        # centerline_geoms_list = [line.simplify(0.2, preserve_topology=True) \
        #                         for line in centerline_geoms_list]
        vectors = self._geom_to_vectors(
            centerline_geoms_list)
        vector_dict.update({'centerline': ('centerline', vectors)})
        return vector_dict

    def union_centerline(self, centerline_geoms):
        # import ipdb;ipdb.set_trace()
        pts_G = nx.DiGraph()
        junction_pts_list = []
        for key, value in centerline_geoms.items():
            centerline_geom = value['centerline']
            if centerline_geom.geom_type == 'MultiLineString':
                start_pt = np.array(centerline_geom.geoms[0].coords).round(3)[0]
                end_pt = np.array(centerline_geom.geoms[-1].coords).round(3)[-1]
                for single_geom in centerline_geom.geoms:
                    single_geom_pts = np.array(single_geom.coords).round(3)
                    for idx, pt in enumerate(single_geom_pts[:-1]):
                        pts_G.add_edge(tuple(single_geom_pts[idx]),tuple(single_geom_pts[idx+1]))
            elif centerline_geom.geom_type == 'LineString':
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]
                for idx, pts in enumerate(centerline_pts[:-1]):
                    pts_G.add_edge(tuple(centerline_pts[idx]),tuple(centerline_pts[idx+1]))
            else:
                raise NotImplementedError
            valid_incoming_num = 0
            for idx, pred in enumerate(value['incoming_tokens']):
                if pred in centerline_geoms.keys():
                    valid_incoming_num += 1
                    pred_geom = centerline_geoms[pred]['centerline']
                    if pred_geom.geom_type == 'MultiLineString':
                        pred_pt = np.array(pred_geom.geoms[-1].coords).round(3)[-1]
        #                 if pred_pt != centerline_pts[0]:
                        pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
                    else:
                        pred_pt = np.array(pred_geom.coords).round(3)[-1]
                        pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
            if valid_incoming_num > 1:
                junction_pts_list.append(tuple(start_pt))
            
            valid_outgoing_num = 0
            for idx, succ in enumerate(value['outgoing_tokens']):
                if succ in centerline_geoms.keys():
                    valid_outgoing_num += 1
                    succ_geom = centerline_geoms[succ]['centerline']
                    if succ_geom.geom_type == 'MultiLineString':
                        succ_pt = np.array(succ_geom.geoms[0].coords).round(3)[0]
        #                 if pred_pt != centerline_pts[0]:
                        pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
                    else:
                        succ_pt = np.array(succ_geom.coords).round(3)[0]
                        pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
            if valid_outgoing_num > 1:
                junction_pts_list.append(tuple(end_pt))

        roots = (v for v, d in pts_G.in_degree() if d == 0)
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            paths = nx.all_simple_paths(pts_G, root, leaves)
            all_paths.extend(paths)

        final_centerline_paths = []
        for path in all_paths:
            merged_line = LineString(path)
            merged_line = merged_line.simplify(0.2, preserve_topology=True)
            final_centerline_paths.append(merged_line)
        return final_centerline_paths, pts_G




def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
    nusc_maps = {}
    map_explorer = {}
    for loc in MAPS:
        nusc_maps[loc] = NuScenesMap(dataroot=root_path, map_name=loc)
        map_explorer[loc] = CNuScenesMapExplorer(nusc_maps[loc])


    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, nusc_maps, map_explorer, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_map_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_map_infos_temporal_train_v3.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_map_infos_temporal_val_v3.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)



def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    # if version == 'v1.0-test':
    #     info_test_path = osp.join(
    #         out_dir, f'{info_prefix}_infos_temporal_test.pkl')
    #     nuscenes_converter.export_2d_annotation(
    #         root_path, info_test_path, version=version)
    # else:
    #     info_train_path = osp.join(
    #         out_dir, f'{info_prefix}_infos_temporal_train.pkl')
    #     info_val_path = osp.join(
    #         out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        # nuscenes_converter.export_2d_annotation(
        #     root_path, info_train_path, version=version)
        # nuscenes_converter.export_2d_annotation(
        #     root_path, info_val_path, version=version)
        # create_groundtruth_database(dataset_name, root_path, info_prefix,
        #                             f'{out_dir}/{info_prefix}_infos_train.pkl')



parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='nuscenes')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()


if __name__ == '__main__':
    train_version = f'{args.version}-trainval'
    nuscenes_data_prep(
        root_path=args.root_path,
        can_bus_root_path=args.canbus,
        info_prefix=args.extra_tag,
        version=train_version,
        dataset_name='NuScenesDataset',
        out_dir=args.out_dir,
        max_sweeps=args.max_sweeps)
    # test_version = f'{args.version}-test'
    # nuscenes_data_prep(
    #     root_path=args.root_path,
    #     can_bus_root_path=args.canbus,
    #     info_prefix=args.extra_tag,
    #     version=test_version,
    #     dataset_name='NuScenesDataset',
    #     out_dir=args.out_dir,
    #     max_sweeps=args.max_sweeps)