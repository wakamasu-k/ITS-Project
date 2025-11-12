
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d
import cv2
import os

# Load dataset
nusc = NuScenes(version='v1.0-mini', dataroot='C:\\Users\\divin\\Downloads\\nuScenes', verbose=True)

# 初期設定
start_sample_number = 79
end_sample_number = 84
#start_sample_number = 202 最初に作成したところ
#end_sample_number = 241
threshold_distance = 2.0  # 2メートルの閾値


transformed_points_list = []

save_folder_camera = "C:\\Users\\divin\\Downloads\\camera"
save_folder_intensity = "C:\\Users\\divin\\Downloads\\intensity"

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
#o3d.io.write_point_cloud("map.pcd", accumulated_pcd)

##########################################################################################################################################################################################

transformed_accumulated_points_list = []

#世界座標系からカメラ座標系への処理

for sample_number in range(start_sample_number, end_sample_number + 1):

    my_sample = nusc.sample[sample_number]

    # カメラデータとLiDARデータを取得
    camera_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
    # キャリブレーションデータを取得
    camera_calibration_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])

    # ego_poseデータを取得
    camera_ego_pose_data = nusc.get('ego_pose', camera_data['ego_pose_token'])

    # カメラの内部キャリブレーションと画像の読み込み
    camera_intrinsic = np.array(camera_calibration_data['camera_intrinsic'])
    camera_image_path = nusc.get_sample_data_path(camera_data['token'])
    camera_image = cv2.imread(camera_image_path)
    image_width = camera_data['width']
    image_height = camera_data['height']


    #世界座標系からカメラ座標系(ego)
    world_cam_ego_transformation_matrix_rt = np.eye(4)  # 4x4の単位行列
    world_cam_ego_transformation_matrix_tr = np.eye(4)  # 4x4の単位行列
    world_cam_ego_transformation_matrix_rt[:3, :3] = Quaternion(camera_ego_pose_data['rotation']).rotation_matrix.T  # 回転行列を挿入
    world_cam_ego_transformation_matrix_tr[:3, 3] = -np.array(camera_ego_pose_data['translation'])  # 並進ベクトルを挿入

    #世界座標系からカメラ座標系(calib)_共通
    world_cam_calib_transformation_matrix_rt = np.eye(4)  # 4x4の単位行列
    world_cam_calib_transformation_matrix_tr = np.eye(4)  # 4x4の単位行列
    world_cam_calib_transformation_matrix_rt[:3, :3] = Quaternion(camera_calibration_data['rotation']).rotation_matrix.T  # 回転行列を挿入
    world_cam_calib_transformation_matrix_tr[:3, 3] = -np.array(camera_calibration_data['translation'])  # 並進ベクトルを挿入

    
    # 結合された点群を個別に変換
    transformed_points = transform_points(accumulated_points, world_cam_ego_transformation_matrix_tr)
    transformed_points = transform_points(transformed_points, world_cam_ego_transformation_matrix_rt)
    transformed_points = transform_points(transformed_points, world_cam_calib_transformation_matrix_tr)
    transformed_points = transform_points(transformed_points, world_cam_calib_transformation_matrix_rt)

    # 変換された点群をリストに保存
    transformed_accumulated_points_list.append(transformed_points)


# 各サンプルの点群を個別に可視化
for transformed_points in transformed_accumulated_points_list:
    # Open3Dでの可視化
    accumulated_pcd = o3d.geometry.PointCloud()
    accumulated_pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])
    # 統合された点群を可視化
    #o3d.visualization.draw_geometries([accumulated_pcd])

# 各サンプルの変換された点群に対する2D描画処理
for i, transformed_points in enumerate(transformed_accumulated_points_list):
    sample_number = start_sample_number + i
    my_sample = nusc.sample[sample_number]

    # カメラデータを取得
    camera_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
    camera_calibration_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])

    # カメラの内部キャリブレーションと画像の読み込み
    camera_intrinsic = np.array(camera_calibration_data['camera_intrinsic'])
    camera_image_path = nusc.get_sample_data_path(camera_data['token'])
    camera_image = cv2.imread(camera_image_path)
    image_width = camera_data['width']
    image_height = camera_data['height']

    # 反射強度を基に色を設定
    reflectance_values = transformed_points[:, 3]
    colors = apply_reflectance_to_colors(reflectance_values)

    # 透視投影行列の作成
    perspective_projection_matrix = np.zeros((3, 4))
    perspective_projection_matrix[:3, :3] = camera_intrinsic

    # 3D点群データから2D点に変換
    points_3d_filtered = transformed_points[transformed_points[:, 2] > 0, :3]#カメラ前方のみにする
    colors_filtered = colors[transformed_points[:, 2] > 0]

    homogeneous_points_3d_filtered = np.hstack((points_3d_filtered, np.ones((points_3d_filtered.shape[0], 1))))
    projected_points_2d_filtered = perspective_projection_matrix @ homogeneous_points_3d_filtered.T
    projected_points_2d_filtered /= projected_points_2d_filtered[2, :]
    projected_points_2d_filtered = projected_points_2d_filtered[:2, :].T


    # 空の画像を作成（全て黒の画像）
    reflectance_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)


    # 点群を距離でソート（遠い点から順に）
    sorting_indices = np.argsort(-points_3d_filtered[:, 2])
    sorted_projected_points_2d = projected_points_2d_filtered[sorting_indices]
    sorted_colors_filtered = colors_filtered[sorting_indices]

    # 画像上に点を描画（遠い点から順）
    for point, color in zip(sorted_projected_points_2d, sorted_colors_filtered):
        x, y = int(point[0]), int(point[1])
        color_bgr = tuple(int(c * 255) for c in color)
        if 0 <= x < image_width and 0 <= y < image_height:
            cv2.circle(reflectance_image, (x, y), 6, color_bgr, -1)
    
    '''
    # 画像を表示（オプション）
    cv2.imshow(f'Intensity Image {sample_number}', reflectance_image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    '''
    # 画像をPNG形式で保存
    save_path = f'Intensity_Image_{sample_number}.png'
    cv2.imwrite(save_path, reflectance_image)
    


    
    '''
    #1例目
    # 各2D座標に対応する最も近い3D点を選択するための辞書
    closest_points = {}

    for point_3d, point_2d, color in zip(points_3d_filtered, projected_points_2d_filtered, colors_filtered):
        x_center, y_center = int(point_2d[0]), int(point_2d[1])
        depth = point_3d[2]  # Z値（深度）

        for x in range(x_center - 3, x_center + 4):
            for y in range(y_center - 3, y_center + 4):
                if 0 <= x < image_width and 0 <= y < image_height:
                    if (x, y) not in closest_points or depth < closest_points[(x, y)][0]:
                        closest_points[(x, y)] = (depth, color)

    # 最も近い点のみを画像に描画
    for (x, y), (_, color) in closest_points.items():
        color_bgr = tuple(int(c * 255) for c in color)
        cv2.circle(reflectance_image, (x, y), 1, color_bgr, -1)


    # 画像を表示（オプション）
    cv2.imshow(f'Intensity Image {sample_number}', reflectance_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    intensity_filename = f'intensity_{sample_number}.jpg'
    intensity_file_path = os.path.join(save_folder_intensity, intensity_filename)
    cv2.imwrite(intensity_filename, reflectance_image)
    '''

    '''
    # 画像上に点を描画
    for point, reflectance in zip(projected_points_2d_filtered, colors_filtered):
        x, y = int(point[0]), int(point[1])
        color_bgr = tuple(int(c * 255) for c in reflectance[::-1])
        if 0 <= x < image_width and 0 <= y < image_height:
            cv2.circle(reflectance_image, (x, y), 4, color_bgr, -1)
    
    # 画像を表示（オプション）
    cv2.imshow(f'Intensity Image {sample_number}', reflectance_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    '''
    intensity_filename = f'intensity_{sample_number}.jpg'
    intensity_file_path = os.path.join(save_folder_intensity, intensity_filename)
    cv2.imwrite(intensity_filename, reflectance_image)
    '''

    '''
    # 2D点を画像に描画
    for point, color in zip(projected_points_2d_filtered, colors_filtered):
        x, y = int(point[0]), int(point[1])
        color_bgr = tuple(int(c * 255) for c in color[::-1])
        if 0 <= x < image_width and 0 <= y < image_height:
            cv2.circle(camera_image, (x, y), 2, color_bgr, -1)
    
    # 画像を表示して保存
    cv2.imshow(f'Camera Image {sample_number}', camera_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    camera_filename = f'camera_{sample_number}.jpg'
    camera_file_path = os.path.join(save_folder_camera, camera_filename)
    cv2.imwrite(camera_filename, camera_image)
    '''
