from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion  # ← 追加
import numpy as np
import open3d as o3d
from tqdm import tqdm

DATA_ROOT = "nuscenes"
nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)

scene = nusc.scene[0]
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

all_points = []

while True:
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    pcl_path = f"{nusc.dataroot}/{lidar_data['filename']}"

    pc = LidarPointCloud.from_file(pcl_path)

    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # ✅ Quaternion へ変換してから行列生成
    cs_trans = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']), inverse=False)
    pose_trans = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']), inverse=False)

    pc.transform(cs_trans)
    pc.transform(pose_trans)

    all_points.append(pc.points.T[:, :3])

    if sample['next'] == "":
        break
    else:
        sample = nusc.get('sample', sample['next'])

all_points = np.concatenate(all_points, axis=0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)

o3d.io.write_point_cloud("nuscenes_map.pcd", pcd)
print("✅ 3Dマップを 'nuscenes_map.pcd' に保存しました")
o3d.visualization.draw_geometries([pcd])
