#NuScenes mini データセットの LiDAR データを指定フレーム範囲で読み込み、車両座標系・世界座標系に変換して統合点群を作るプログラム
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d
import cv2
import os

# Load dataset
nusc = NuScenes(
    version='v1.0-mini',
    dataroot='D:\\Users\\wakamatsu.k\\Desktop\\ITS\\nuscenes',
    verbose=True
)
# 初期設定
start_sample_number = 364
end_sample_number = 403
#start_sample_number = 202 最初に作成したところ
#end_sample_number = 241
threshold_distance = 2.0  # 2メートルの閾値


transformed_points_list = []

save_folder_camera = "nuscenes\camera"
save_folder_intensity = "nuscenes\intensity"

# LiDARデータを読み込み
def load_lidar_data(filepath):
    pc = np.fromfile(filepath, dtype=np.float32)
    pc = pc.reshape(-1, 5)  # XYZ座標、反射強度、タイムスタンプ
    return pc[:, :4]  # 最初の4列（XYZ座標と反射強度）を取得

# 点群を変換する関数
def transform_points(points, transformation_matrix):
    points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    points_transformed = np.dot(transformation_matrix, points_homogeneous.T).T
    return np.hstack((points_transformed[:, :3], points[:, 3:4]))  # XYZ座標と反射強度


def apply_reflectance_to_colors(reflectance):
    """
    反射強度を基に色を設定する関数。反射強度はグレースケールの色に変換される。
    """
    # 反射強度を0から1の範囲に正規化するための倍率を調整
    max_reflectance_factor = 0.5  # 最大値の倍率を下げる
    min_reflectance_factor = 1.2  # 最小値の倍率を上げる

    # 反射強度を調整した範囲で正規化
    max_reflectance = np.max(reflectance) * max_reflectance_factor
    min_reflectance = np.min(reflectance) * min_reflectance_factor
    reflectance_normalized = (reflectance - min_reflectance) / (max_reflectance - min_reflectance)

    # クリッピングして範囲を0から1に制限
    reflectance_normalized = np.clip(reflectance_normalized, 0, 1)

    # カラー表現への変換
    colors = np.vstack([reflectance_normalized, reflectance_normalized, reflectance_normalized]).T
    return colors


##########################################################################################################################################################################################

# センサー座標系→車両座標系→世界座標系の処理と点群の統合
for sample_number in range(start_sample_number, end_sample_number + 1):

    my_sample = nusc.sample[sample_number]

    lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])

    lidar_calibration_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    lidar_ego_pose_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # 変換行列の計算
    sensor_vehicle_matrix_rt = np.eye(4)
    sensor_vehicle_matrix_rt[:3, :3] = Quaternion(lidar_calibration_data['rotation']).rotation_matrix
    sensor_vehicle_matrix_tr = np.eye(4)
    sensor_vehicle_matrix_tr[:3, 3] = np.array(lidar_calibration_data['translation'])

    vehicle_world_matrix_rt = np.eye(4)
    vehicle_world_matrix_rt[:3, :3] = Quaternion(lidar_ego_pose_data['rotation']).rotation_matrix
    vehicle_world_matrix_tr = np.eye(4)
    vehicle_world_matrix_tr[:3, 3] = np.array(lidar_ego_pose_data['translation'])

    # LiDARデータの読み込みと変換
    lidar_filepath = nusc.get_sample_data_path(lidar_data['token'])
    lidar_points = load_lidar_data(lidar_filepath)
    distances_xy = np.linalg.norm(lidar_points[:, :2], axis=1)
    mask = distances_xy > threshold_distance
    lidar_points_filtered = lidar_points[mask]

    lidar_points_transformed = transform_points(lidar_points_filtered, sensor_vehicle_matrix_rt)
    lidar_points_transformed = transform_points(lidar_points_transformed, sensor_vehicle_matrix_tr)
    lidar_points_transformed = transform_points(lidar_points_transformed, vehicle_world_matrix_rt)
    lidar_points_transformed = transform_points(lidar_points_transformed, vehicle_world_matrix_tr)


    transformed_points_list.append(lidar_points_transformed)

# すべての変換された点群を結合
accumulated_points = np.vstack(transformed_points_list)

# Open3Dでの可視化
accumulated_pcd = o3d.geometry.PointCloud()
accumulated_pcd.points = o3d.utility.Vector3dVector(accumulated_points[:, :3])
# 統合された点群を可視化
o3d.visualization.draw_geometries([accumulated_pcd])
# 点群マップの保存
o3d.io.write_point_cloud("map.pcd", accumulated_pcd)

##########################################################################################################################################################################################
