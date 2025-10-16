from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon
from shapely import affinity

class CNuScenesMapExplorer(NuScenesMapExplorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_max_drivable_surface(self, patch_box, patch_angle, location):
        """
        Get the maximum drivable surface within a specified patch.

        Args:
            patch_box (tuple): (x_center, y_center, width, height) of the patch.
            patch_angle (float): Rotation angle of the patch in degrees.
            location (str): Location name (e.g., 'boston-seaport').

        Returns:
            MultiPolygon: Maximum drivable surface within the patch.
        """
        patch_x, patch_y, patch_width, patch_height = patch_box
        patch = self.get_patch_coord(patch_box, patch_angle)

        records = self.map_api.drivable_area
        polygon_list = []

        for record in records:
            polygons = [self.map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
            for polygon in polygons:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        polygon_list.append(new_polygon)
                    elif new_polygon.geom_type == 'MultiPolygon':
                        polygon_list.extend(new_polygon)

        return MultiPolygon(polygon_list)
# Укажите путь к данным NuScenes
root_path = '/media/livanoff/Новый том2/PycharmProjects/nuscenes'
loc = 'boston-seaport'  # Замените на нужную карту
point_cloud_range=[-15.0, -30.0,-10.0, 15.0, 30.0, 10.0]
# Загрузка карты
nusc_maps = {}
map_explorer = {}
nusc = NuScenes(version='v1.0-trainval', dataroot=root_path, verbose=True)
nusc_maps[loc] = NuScenesMap(dataroot=root_path, map_name=loc)
map_explorer[loc] = CNuScenesMapExplorer(nusc_maps[loc])

for sample in nusc.sample:
    map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']

    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor',
                        sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

    info = {
                'lidar_path': lidar_path,
                'token': sample['token'],
                'prev': sample['prev'],
                'next': sample['next'],
                'frame_idx': 0,  # temporal related info
                'sweeps': [],
                'cams': dict(),
                'map_location': map_location,
                'scene_token': sample['scene_token'],  # temporal related info
                'lidar2ego_translation': cs_record['translation'],
                'lidar2ego_rotation': cs_record['rotation'],
                'ego2global_translation': pose_record['translation'],
                'ego2global_rotation': pose_record['rotation'],
                'timestamp': sample['timestamp'],
            }
    import numpy as np
    from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
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
    
    map_pose = lidar2global_translation[:2]
    rotation = Quaternion(lidar2global_rotation)
    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    surface = map_explorer[loc].get_max_drivable_surface(patch_box, patch_angle, location)
    # Отображение карты
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if len(surface) == 0:
        continue
    
    fig, ax = plt.subplots()
    from descartes import PolygonPatch
    for polygon in surface:
        patch = PolygonPatch(polygon, facecolor='blue', edgecolor='blue', alpha=0.5, zorder=2)
        ax.add_patch(patch)

    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_aspect('equal', 'box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Drivable Surface')
    plt.show()