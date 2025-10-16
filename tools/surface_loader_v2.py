import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
from shapely import affinity
from descartes import PolygonPatch
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
import numpy as np
import cv2

INTERPOLATION = cv2.LINE_8
import torch
import numpy as np
import cv2

from pathlib import Path
from functools import lru_cache

from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon


STATIC = ['ped_crossing', 'drivable_area', 'road_segment']
CLASSES = STATIC
NUM_CLASSES = len(CLASSES)
INTERPOLATION = cv2.LINE_8

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
    import os
    import mmcv
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

def get_split(split, dataset_name):
    split_dir = Path(__file__).parent / 'splits' / dataset_name
    split_path = split_dir / f'{split}.txt'

    return split_path.read_text().strip().split('\n')


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return np.float32([
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ])


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

def get_data(
    dataset_dir,
    split,
    version,           # ignore
    num_classes=NUM_CLASSES,            # in here to make config consistent
    **dataset_kwargs
):
    assert num_classes == NUM_CLASSES
    
    helper = NuScenesSingleton(dataset_dir, version)
    
    from nuscenes.utils import splits
    train_scenes = splits.train
    val_scenes = splits.val
    available_scenes = get_available_scenes(helper.nusc)
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

    transform = None
    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    # split_scenes = get_split(split, 'nuscenes')

    for scene_name, scene_record in helper.get_scenes():
        if scene_record['token'] in val_scenes:

            data = NuScenesDataset(scene_name, scene_record, helper,
                                transform=transform, **dataset_kwargs)
            # for i in range(data.__len__()):
            #     data.__getitem__(i)
            data.__getitem__(0)


class NuScenesSingleton:
    """
    Wraps both nuScenes and nuScenes map API

    This was an attempt to sidestep the 30 second loading time in a "clean" manner
    """
    def __init__(self, dataset_dir, version):
        """
        dataset_dir: /path/to/nuscenes/
        version: v1.0-trainval
        """
        self.dataroot = str(dataset_dir)
        self.nusc = self.lazy_nusc(version, self.dataroot)

    @classmethod
    def lazy_nusc(cls, version, dataroot):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.nuscenes import NuScenes

        if not hasattr(cls, '_lazy_nusc'):
            cls._lazy_nusc = NuScenes(version=version, dataroot=dataroot)

        return cls._lazy_nusc

    def get_scenes(self):
        for scene_record in self.nusc.scene:
            yield scene_record['name'], scene_record

    @lru_cache(maxsize=16)
    def get_map(self, log_token):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.map_expansion.map_api import NuScenesMap

        map_name = self.nusc.get('log', log_token)['location']
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        return nusc_map

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            obj = super(NuScenesSingleton, cls).__new__(cls)
            obj.__init__(*args, **kwargs)

            cls._singleton = obj

        return cls._singleton


class NuScenesDataset(torch.utils.data.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def __init__(
        self,
        scene_name: str,
        scene_record: dict,
        helper: NuScenesSingleton,
        transform=None,
        cameras=[[0, 1, 2, 3, 4, 5]],
        bev={'h': 200, 'w': 100, 'h_meters': 60, 'w_meters': 30, 'offset': 0.0},
    ):
        self.scene_name = scene_name
        self.transform = transform

        self.nusc = helper.nusc
        self.nusc_map = helper.get_map(scene_record['log_token'])

        self.view = get_view_matrix(**bev)
        self.bev_shape = (bev['h'], bev['w'])

        self.samples = self.parse_scene(scene_record, cameras)

    def parse_scene(self, scene_record, camera_rigs):
        data = []
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = self.nusc.get('sample', sample_token)

            for camera_rig in camera_rigs:
                data.append(self.parse_sample_record(sample_record, camera_rig))

            sample_token = sample_record['next']

        return data

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def parse_sample_record(self, sample_record, camera_rig):
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        lidar_path = lidar_record['filename']
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])

        world_from_egolidarflat = self.parse_pose(egolidar, flat=True)
        egolidarflat_from_world = self.parse_pose(egolidar, flat=True, inv=True)
        
        sd_rec = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
        
        
        cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []

        for cam_idx in camera_rig:
            cam_channel = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_channel]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            cam_from_egocam = self.parse_pose(cam, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)

            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            image_path = str(full_path.relative_to(self.nusc.dataroot))

            cam_channels.append(cam_channel)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)

        return {
            'scene': self.scene_name,
            'token': sample_record['token'],

            'pose': world_from_egolidarflat.tolist(),
            'pose_inverse': egolidarflat_from_world.tolist(),
            
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],

            'cam_ids': list(camera_rig),
            'cam_channels': cam_channels,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'images': images,
            'lidar_path': lidar_path
        }


    def get_static_layers(self, sample, layers, point_cloud_range = [-10.0, -10.0,-10.0, 10.0, 10.0, 10.0]):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
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
        map_pose = lidar2global_translation[:2]
        
        patch_w = point_cloud_range[3] - point_cloud_range[0]
        patch_h = point_cloud_range[4] - point_cloud_range[1]
        patch_size = (patch_h, patch_w)
        patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
        records_in_patch = self.nusc_map.get_records_in_patch(patch_box, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)

                if layer == 'drivable_area': polygon_tokens = polygon_token['polygon_tokens']
                else: polygon_tokens = [polygon_token['polygon_token']]

                for p in polygon_tokens:
                    polygon = self.nusc_map.extract_polygon(p)
                    polygon = MultiPolygon([polygon])

                    exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in exteriors]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in exteriors]
                    exteriors = [V @ S @ M_inv @ p for p in exteriors]
                    exteriors = [p[:2].round().astype(np.int32).T for p in exteriors]

                    cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

                    interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in interiors]
                    interiors = [V @ S @ M_inv @ p for p in interiors]
                    interiors = [p[:2].round().astype(np.int32).T for p in interiors]

                    cv2.fillPoly(render, interiors, 0, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        bev = self.get_static_layers(sample, STATIC)        

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))

        # Первый канал (lane)
        plt.subplot(1, 2, 1)
        plt.imshow(bev[:, :, 0], cmap='gray')
        plt.title("Lane Layer")

        # Второй канал (road_segment)
        plt.subplot(1, 2, 2)
        plt.imshow(bev[:, :, 1], cmap='gray')
        plt.title("Road Segment Layer")
        print(sample['lidar_path'])
        plt.show()

        
        return bev

# Укажите путь к данным NuScenes
root_path = '/media/livanoff/Новый том2/PycharmProjects/nuscenes'
loc = 'boston-seaport'  # Замените на нужную карту
point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
patch_w = point_cloud_range[3] - point_cloud_range[0]
patch_h = point_cloud_range[4] - point_cloud_range[1]
patch_size = (patch_h, patch_w)

get_data(root_path, 'test', 'v1.0-trainval')


# for sample in nusc.sample:
#     map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']

#     lidar_token = sample['data']['LIDAR_TOP']
#     sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
#     cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
#     pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
#     lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

#     info = {
#         'lidar_path': lidar_path,
#         'token': sample['token'],
#         'prev': sample['prev'],
#         'next': sample['next'],
#         'frame_idx': 0,  # temporal related info
#         'sweeps': [],
#         'cams': dict(),
#         'map_location': map_location,
#         'scene_token': sample['scene_token'],  # temporal related info
#         'lidar2ego_translation': cs_record['translation'],
#         'lidar2ego_rotation': cs_record['rotation'],
#         'ego2global_translation': pose_record['translation'],
#         'ego2global_rotation': pose_record['rotation'],
#         'timestamp': sample['timestamp'],
#     }

    

#     map_pose = pose_record['translation'][:2]
#     rotation = Quaternion(pose_record['rotation'])
#     translation = np.array(pose_record['translation'])
#     ego_translation = np.array(pose_record['translation'])[:2]
     
#     patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
#     patch_angle = quaternion_yaw(rotation) / np.pi * 180
    
#     M = np.eye(4)
#     M[:3, :3] = rotation.rotation_matrix
#     M[:3, 3] = translation

#     # Получаем обратную матрицу позы
#     M_inv = np.linalg.inv(M)

#     print(f"Patch Box: {patch_box}, Patch Angle: {patch_angle}")

#     layers = ['lane', 'road_segment']
#     surface = map_explorer[loc].get_static_layers(sample, layers, M_inv, ego_translation)
#     if surface is None:
#         print("No drivable surface found.")
#         continue

#     print(f"Center: ({map_pose[0]}, {map_pose[1]})")