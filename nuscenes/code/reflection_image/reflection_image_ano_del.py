#nuScenes の LiDAR 点群データを使って「静的3Dマップ」を生成し、カメラ画像上に点群を投影して可視化・保存する
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d
import cv2
import os

# =========================================================
# 1. Boxの中に点が含まれているかどうか判定するユーティリティ関数
# =========================================================
def points_in_box_3d(box: Box, points_xyz: np.ndarray) -> np.ndarray:
    center = box.center           # [cx, cy, cz]
    orientation = box.orientation # pyquaternion.Quaternion
    w, l, h = box.wlh

    # 1) 点群を box の中心へ平行移動
    #    shape=(N,3)
    points_shifted = points_xyz - center[None, :]

    # 2) 逆回転行列 (R_inv) を用いて一括回転
    R_inv = orientation.inverse.rotation_matrix  # shape=(3,3)
    points_in_box_coords = points_shifted @ R_inv.T  # shape=(N,3)

    # 3) 軸平行AABB内チェック
    half_w, half_l, half_h = w/2.0, l/2.0, h/2.0
    mask_x = np.abs(points_in_box_coords[:, 0]) <= half_w
    mask_y = np.abs(points_in_box_coords[:, 1]) <= half_l
    mask_z = np.abs(points_in_box_coords[:, 2]) <= half_h
    mask_in_box = mask_x & mask_y & mask_z

    return mask_in_box


def is_dynamic_category(category_name: str) -> bool:
    """
    動的物体とみなすカテゴリを指定。
    ここでは vehicle.*, human.*, movable_object.* を除去したい場合の例。
    必要に応じて条件を拡張or変更。
    """
    top_level = category_name.split('.')[0]
    return top_level in {'vehicle', 'human', 'movable_object'}


# =========================================================
# 2. メインコード
# =========================================================

# Load dataset
nusc = NuScenes(version='v1.0-mini', dataroot='D:\\Users\\wakamatsu.k\\Desktop\\ITS\\nuscenes', verbose=True)

# 初期設定
start_sample_number = 39
end_sample_number = 78
threshold_distance = 2.0  # 2メートルの閾値

transformed_points_list = []

save_folder_camera = "nuscenes\camera" #カメラ画像を保存
save_folder_intensity = "nuscenes\intensity"#LIDERの反射強度を保存


# LiDARデータを読み込み
def load_lidar_data(filepath):
    pc = np.fromfile(filepath, dtype=np.float32)
    pc = pc.reshape(-1, 5)  # XYZ座標、反射強度、タイムスタンプ
    return pc[:, :4]  # 最初の4列（XYZ座標と反射強度）を取得

# 点群を4x4行列で変換する関数
def transform_points(points, transformation_matrix):
    points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    points_transformed = np.dot(transformation_matrix, points_homogeneous.T).T
    return np.hstack((points_transformed[:, :3], points[:, 3:4]))  # XYZ座標と反射強度

def apply_reflectance_to_colors(reflectance):
    """
    反射強度を基に色を設定する関数。反射強度はグレースケールの色に変換される。
    """
    max_reflectance_factor = 0.5
    min_reflectance_factor = 1.2

    max_reflectance = np.max(reflectance) * max_reflectance_factor
    min_reflectance = np.min(reflectance) * min_reflectance_factor
    reflectance_normalized = (reflectance - min_reflectance) / (max_reflectance - min_reflectance)
    reflectance_normalized = np.clip(reflectance_normalized, 0, 1)

    colors = np.vstack([reflectance_normalized, reflectance_normalized, reflectance_normalized]).T
    return colors


# =====================================================================
# センサー座標系→車両座標系→世界座標系の処理と点群の統合 (動的物体除去を追加)
# =====================================================================
for sample_number in range(start_sample_number, end_sample_number + 1):

    my_sample = nusc.sample[sample_number]
    lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])

    # LiDARのキャリブレーション・ego_pose
    lidar_calibration_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_ego_pose_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # 変換行列
    sensor_vehicle_matrix_rt = np.eye(4)
    sensor_vehicle_matrix_rt[:3, :3] = Quaternion(lidar_calibration_data['rotation']).rotation_matrix
    sensor_vehicle_matrix_tr = np.eye(4)
    sensor_vehicle_matrix_tr[:3, 3] = np.array(lidar_calibration_data['translation'])

    vehicle_world_matrix_rt = np.eye(4)
    vehicle_world_matrix_rt[:3, :3] = Quaternion(lidar_ego_pose_data['rotation']).rotation_matrix
    vehicle_world_matrix_tr = np.eye(4)
    vehicle_world_matrix_tr[:3, 3] = np.array(lidar_ego_pose_data['translation'])

    # LiDARデータの読み込み
    lidar_filepath = nusc.get_sample_data_path(lidar_data['token'])
    lidar_points = load_lidar_data(lidar_filepath)

    # 近すぎる点を除外
    distances_xy = np.linalg.norm(lidar_points[:, :2], axis=1)
    mask = distances_xy > threshold_distance
    lidar_points_filtered = lidar_points[mask]

    # センサー座標 -> 車両座標 -> 世界座標へ変換
    lidar_points_transformed = transform_points(lidar_points_filtered, sensor_vehicle_matrix_rt)
    lidar_points_transformed = transform_points(lidar_points_transformed, sensor_vehicle_matrix_tr)
    lidar_points_transformed = transform_points(lidar_points_transformed, vehicle_world_matrix_rt)
    lidar_points_transformed = transform_points(lidar_points_transformed, vehicle_world_matrix_tr)

    # -----------------------------
    # (追加) 動的物体を除去する処理
    # -----------------------------
    # 現フレームのアノテーションBoxを取得 (世界座標系)
    boxes = nusc.get_boxes(lidar_data['token'])

    # 動的カテゴリなら、そのBox内の点を除去
    for box in boxes:
        if is_dynamic_category(box.name):
            mask_in_box = points_in_box_3d(box, lidar_points_transformed[:, :3])
            lidar_points_transformed = lidar_points_transformed[~mask_in_box]

    # 結果をリストに追加
    transformed_points_list.append(lidar_points_transformed)

# すべての変換された点群を結合
accumulated_points = np.vstack(transformed_points_list)

# Open3Dでの可視化 (動的物体除去後の地図)
accumulated_pcd = o3d.geometry.PointCloud()
accumulated_pcd.points = o3d.utility.Vector3dVector(accumulated_points[:, :3])
o3d.visualization.draw_geometries([accumulated_pcd])
# 必要なら保存
# o3d.io.write_point_cloud("map_static_only.pcd", accumulated_pcd)

# =====================================================================
# 以下、世界座標系からカメラ座標系への変換 (既存の処理)
# =====================================================================
transformed_accumulated_points_list = []

for sample_number in range(start_sample_number, end_sample_number + 1):

    my_sample = nusc.sample[sample_number]

    camera_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
    camera_calibration_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
    camera_ego_pose_data = nusc.get('ego_pose', camera_data['ego_pose_token'])

    camera_intrinsic = np.array(camera_calibration_data['camera_intrinsic'])
    camera_image_path = nusc.get_sample_data_path(camera_data['token'])
    camera_image = cv2.imread(camera_image_path)
    image_width = camera_data['width']
    image_height = camera_data['height']

    # 世界座標 -> カメラego座標
    world_cam_ego_rt = np.eye(4)
    world_cam_ego_tr = np.eye(4)
    world_cam_ego_rt[:3, :3] = Quaternion(camera_ego_pose_data['rotation']).rotation_matrix.T
    world_cam_ego_tr[:3, 3] = -np.array(camera_ego_pose_data['translation'])

    # 世界座標 -> カメラキャリブ座標
    world_cam_calib_rt = np.eye(4)
    world_cam_calib_tr = np.eye(4)
    world_cam_calib_rt[:3, :3] = Quaternion(camera_calibration_data['rotation']).rotation_matrix.T
    world_cam_calib_tr[:3, 3] = -np.array(camera_calibration_data['translation'])

    # 既に動的物体除去済みのaccumulated_pointsを変換
    transformed_points = transform_points(accumulated_points, world_cam_ego_tr)
    transformed_points = transform_points(transformed_points, world_cam_ego_rt)
    transformed_points = transform_points(transformed_points, world_cam_calib_tr)
    transformed_points = transform_points(transformed_points, world_cam_calib_rt)

    transformed_accumulated_points_list.append(transformed_points)

# 以降は2D投影の可視化・PNG保存 (元のコード同様)
for i, transformed_points in enumerate(transformed_accumulated_points_list):
    sample_number = start_sample_number + i
    my_sample = nusc.sample[sample_number]

    camera_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
    camera_calibration_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
    camera_intrinsic = np.array(camera_calibration_data['camera_intrinsic'])
    camera_image_path = nusc.get_sample_data_path(camera_data['token'])
    camera_image = cv2.imread(camera_image_path)
    image_width = camera_data['width']
    image_height = camera_data['height']

    # 反射強度を基に色を設定
    reflectance_values = transformed_points[:, 3]
    colors = apply_reflectance_to_colors(reflectance_values)

    # 透視投影行列
    perspective_projection_matrix = np.zeros((3, 4))
    perspective_projection_matrix[:3, :3] = camera_intrinsic

    # カメラ前方の点だけ取り出し
    points_3d_filtered = transformed_points[transformed_points[:, 2] > 0, :3]
    colors_filtered = colors[transformed_points[:, 2] > 0]

    homogeneous_points_3d_filtered = np.hstack((points_3d_filtered, np.ones((points_3d_filtered.shape[0], 1))))
    projected_points_2d_filtered = perspective_projection_matrix @ homogeneous_points_3d_filtered.T
    projected_points_2d_filtered /= projected_points_2d_filtered[2, :]
    projected_points_2d_filtered = projected_points_2d_filtered[:2, :].T

    # 空の黒画像
    reflectance_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # 距離が遠い順にソートして描画
    sorting_indices = np.argsort(-points_3d_filtered[:, 2])
    sorted_projected_points_2d = projected_points_2d_filtered[sorting_indices]
    sorted_colors_filtered = colors_filtered[sorting_indices]

    for point, color in zip(sorted_projected_points_2d, sorted_colors_filtered):
        x, y = int(point[0]), int(point[1])
        color_bgr = tuple(int(c * 255) for c in color)
        if 0 <= x < image_width and 0 <= y < image_height:
            cv2.circle(reflectance_image, (x, y), 1, color_bgr, -1)

    # 画像をPNG形式で保存
    save_path = os.path.join(save_folder_intensity, f"Intensity_Image_{sample_number}.png")
    cv2.imwrite(save_path, reflectance_image)

