from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import open3d as o3d
import os

# === データセットの読み込み ===
nusc = NuScenes(version='v1.0-mini', dataroot='nuscenes', verbose=True)

# 最初のシーンのLiDARフレームを取得
scene = nusc.scene[0]
sample_token = scene['first_sample_token']
sample = nusc.get('sample', sample_token)

# LIDAR_TOPデータを取得
lidar_token = sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)

# 点群の読み込み
lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])
pc = LidarPointCloud.from_file(lidar_path)
print(pc.points.shape)  # 形状: (4, N) → x,y,z,反射強度
