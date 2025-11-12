from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d
import cv2

# Load dataset
nusc = NuScenes(version='v1.0-mini', dataroot='nuScenes', verbose=True)

my_sample_202 = nusc.sample[202]
my_sample_203 = nusc.sample[203]
my_sample_204 = nusc.sample[204]

# カメラデータとLiDARデータを取得
camera_data_202 = nusc.get('sample_data', my_sample_202['data']['CAM_FRONT'])
camera_data_203 = nusc.get('sample_data', my_sample_203['data']['CAM_FRONT'])
camera_data_204 = nusc.get('sample_data', my_sample_204['data']['CAM_FRONT'])

lidar_data_202 = nusc.get('sample_data', my_sample_202['data']['LIDAR_TOP'])
lidar_data_203 = nusc.get('sample_data', my_sample_203['data']['LIDAR_TOP'])
lidar_data_204 = nusc.get('sample_data', my_sample_204['data']['LIDAR_TOP'])

# カメラデータとLiDARセンサーのキャリブレーションデータを取得
camera_calibration_data_202 = nusc.get('calibrated_sensor', camera_data_202['calibrated_sensor_token'])
camera_calibration_data_203 = nusc.get('calibrated_sensor', camera_data_203['calibrated_sensor_token'])
camera_calibration_data_204 = nusc.get('calibrated_sensor', camera_data_204['calibrated_sensor_token'])

lidar_calibration_data_202 = nusc.get('calibrated_sensor', lidar_data_202['calibrated_sensor_token'])
lidar_calibration_data_203 = nusc.get('calibrated_sensor', lidar_data_203['calibrated_sensor_token'])
lidar_calibration_data_204 = nusc.get('calibrated_sensor', lidar_data_204['calibrated_sensor_token'])

# フレームごとのego_poseデータを取得
camera_ego_pose_data_202 = nusc.get('ego_pose', camera_data_202['ego_pose_token'])
camera_ego_pose_data_203 = nusc.get('ego_pose', camera_data_203['ego_pose_token'])
camera_ego_pose_data_204 = nusc.get('ego_pose', camera_data_204['ego_pose_token'])

lidar_ego_pose_data_202 = nusc.get('ego_pose', lidar_data_202['ego_pose_token'])
lidar_ego_pose_data_203 = nusc.get('ego_pose', lidar_data_203['ego_pose_token'])
lidar_ego_pose_data_204 = nusc.get('ego_pose', lidar_data_204['ego_pose_token'])

# カメラの内部キャリブレーション(共通)
camera_intrinsic_202 = np.array(camera_calibration_data_202['camera_intrinsic'])


# カメラ画像を読み込む
camera_image_path_202 = nusc.get_sample_data_path(camera_data_202['token'])
camera_image_202 = cv2.imread(camera_image_path_202)


# 画像の寸法を取得
image_width_202 = camera_data_202['width']
image_height_202 = camera_data_202['height']


############################################################################### センサー座標系から車両座標系の変換行列

# キャリブレーションのrotationとtranslation, camera_intrinsicは共通なので202が代表
# クォータニオンから回転行列を作成
sensor_vehicle_quaternion_202 = Quaternion(lidar_calibration_data_202['rotation'])
sensor_vehicle_translation_202 = np.array(lidar_calibration_data_202['translation'])

sensor_vehicle_transformation_matrix_202_rt = np.eye(4)  # 4x4の単位行列
sensor_vehicle_transformation_matrix_202_tr = np.eye(4)  # 4x4の単位行列
sensor_vehicle_transformation_matrix_202_rt[:3, :3] = sensor_vehicle_quaternion_202.rotation_matrix  # 回転行列を挿入
sensor_vehicle_transformation_matrix_202_tr[:3, 3] = sensor_vehicle_translation_202  # 並進ベクトルを挿入


vehicle_camera_transformation_matrix_202_rt = np.eye(4)  # 4x4の単位行列
vehicle_camera_transformation_matrix_202_tr = np.eye(4)  # 4x4の単位行列
vehicle_camera_transformation_matrix_202_rt[:3, :3] = Quaternion(camera_calibration_data_202['rotation']).rotation_matrix.T  # 回転行列を挿入
vehicle_camera_transformation_matrix_202_tr[:3, 3] = -np.array(camera_calibration_data_202['translation'])  # 並進ベクトルを挿入


############################################################################### 車両座標系から世界座標系の変換行列


# 202
# 回転（クォータニオン）と並進（位置）を取得
vehicle_world_rotation_202 = Quaternion(lidar_ego_pose_data_202['rotation'])
vehicle_world_translation_202 = np.array(lidar_ego_pose_data_202['translation'])


# x vehicle, o lidar
vehicle_world_transformation_matrix_202_rt = np.eye(4)  # 4x4の単位行列
vehicle_world_transformation_matrix_202_tr = np.eye(4)  # 4x4の単位行列
vehicle_world_transformation_matrix_202_rt[:3, :3] = vehicle_world_rotation_202.rotation_matrix  # 回転行列を挿入
vehicle_world_transformation_matrix_202_tr[:3, 3] = vehicle_world_translation_202  # 並進ベクトルを挿入

world_cam_transformation_matrix_202_rt = np.eye(4)  # 4x4の単位行列
world_cam_transformation_matrix_202_tr = np.eye(4)  # 4x4の単位行列
world_cam_transformation_matrix_202_rt[:3, :3] = Quaternion(camera_ego_pose_data_202['rotation']).rotation_matrix.T  # 回転行列を挿入
world_cam_transformation_matrix_202_tr[:3, 3] = -np.array(camera_ego_pose_data_202['translation'])  # 並進ベクトルを挿入


# 203
# 回転（クォータニオン）と並進（位置）を取得
vehicle_world_rotation_203 = Quaternion(lidar_ego_pose_data_203['rotation'])
vehicle_world_translation_203 = np.array(lidar_ego_pose_data_203['translation'])

vehicle_world_transformation_matrix_203_rt = np.eye(4)  # 4x4の単位行列
vehicle_world_transformation_matrix_203_tr = np.eye(4)  # 4x4の単位行列
vehicle_world_transformation_matrix_203_rt[:3, :3] = vehicle_world_rotation_203.rotation_matrix  # 回転行列を挿入
vehicle_world_transformation_matrix_203_tr[:3, 3] = vehicle_world_translation_203  # 並進ベクトルを挿入

# 204
# 回転（クォータニオン）と並進（位置）を取得
vehicle_world_rotation_204 = Quaternion(lidar_ego_pose_data_204['rotation'])
vehicle_world_translation_204 = np.array(lidar_ego_pose_data_204['translation'])


vehicle_world_transformation_matrix_204_rt = np.eye(4)  # 4x4の単位行列
vehicle_world_transformation_matrix_204_tr = np.eye(4)  # 4x4の単位行列
vehicle_world_transformation_matrix_204_rt[:3, :3] = vehicle_world_rotation_204.rotation_matrix  # 回転行列を挿入
vehicle_world_transformation_matrix_204_tr[:3, 3] = vehicle_world_translation_204  # 並進ベクトルを挿入



#####################################################################################変換行列の適応と点群の結合

# LiDARデータを読み込み
def load_lidar_data(filepath):

    pc = np.fromfile(filepath, dtype=np.float32)
    pc = pc.reshape(-1, 5)  # XYZ座標、反射強度、タイムスタンプ
    return pc[:, :4]  # 最初の4列（XYZ座標と反射強度）を取得強度

def transform_points(points, transformation_matrix):
    points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    points_transformed = np.dot(transformation_matrix, points_homogeneous.T).T
    return np.hstack((points_transformed[:, :3], points[:, 3:4]))  # XYZ座標と反射強度

# ensor_vehicle_transformation_matrix_202は203でも204でも同値なため202のみ使用
# 202
# 各サンプルのLiDARデータを読み込み、変換行列を適用
lidar_filepath_202 = nusc.get_sample_data_path(lidar_data_202['token'])
lidar_points_202 = load_lidar_data(lidar_filepath_202)
#lidar_points_202_transformed = transform_points(lidar_points_202, sensor_vehicle_transformation_matrix_202)
#lidar_points_202_transformed = transform_points(lidar_points_202_transformed, vehicle_world_transformation_matrix_202)
#lidar_points_202_transformed = transform_points(lidar_points_202_transformed, view_matrix_202)

lidar_points_202_transformed = transform_points(lidar_points_202, sensor_vehicle_transformation_matrix_202_rt)
lidar_points_202_transformed = transform_points(lidar_points_202_transformed, sensor_vehicle_transformation_matrix_202_tr)
lidar_points_202_transformed = transform_points(lidar_points_202_transformed, vehicle_world_transformation_matrix_202_rt)
lidar_points_202_transformed = transform_points(lidar_points_202_transformed, vehicle_world_transformation_matrix_202_tr)


# 203
# 各サンプルのLiDARデータを読み込み、変換行列を適用
lidar_filepath_203 = nusc.get_sample_data_path(lidar_data_203['token'])
lidar_points_203 = load_lidar_data(lidar_filepath_203)
#lidar_points_203_transformed = transform_points(lidar_points_203, sensor_vehicle_transformation_matrix_202)
#lidar_points_203_transformed = transform_points(lidar_points_203_transformed, vehicle_world_transformation_matrix_203)
#lidar_points_203_transformed = transform_points(lidar_points_203_transformed, view_matrix_203)

lidar_points_203_transformed = transform_points(lidar_points_203, sensor_vehicle_transformation_matrix_202_rt)
lidar_points_203_transformed = transform_points(lidar_points_203_transformed, sensor_vehicle_transformation_matrix_202_tr)
lidar_points_203_transformed = transform_points(lidar_points_203_transformed, vehicle_world_transformation_matrix_203_rt)
lidar_points_203_transformed = transform_points(lidar_points_203_transformed, vehicle_world_transformation_matrix_203_tr)


# 204
# 各サンプルのLiDARデータを読み込み、変換行列を適用

lidar_filepath_204 = nusc.get_sample_data_path(lidar_data_204['token'])
lidar_points_204 = load_lidar_data(lidar_filepath_204)
#lidar_points_204_transformed = transform_points(lidar_points_204, sensor_vehicle_transformation_matrix_202)
#lidar_points_204_transformed = transform_points(lidar_points_204_transformed, vehicle_world_transformation_matrix_204)
#lidar_points_204_transformed = transform_points(lidar_points_204_transformed, view_matrix_204)

lidar_points_204_transformed = transform_points(lidar_points_204, sensor_vehicle_transformation_matrix_202_rt)
lidar_points_204_transformed = transform_points(lidar_points_204_transformed, sensor_vehicle_transformation_matrix_202_tr)
lidar_points_204_transformed = transform_points(lidar_points_204_transformed, vehicle_world_transformation_matrix_204_rt)
lidar_points_204_transformed = transform_points(lidar_points_204_transformed, vehicle_world_transformation_matrix_204_tr)

accumulated_points = np.vstack([lidar_points_202_transformed, lidar_points_203_transformed, lidar_points_204_transformed])
#accumulated_points = np.vstack([lidar_points_202_transformed, lidar_points_203_transformed])
#accumulated_points = lidar_points_202_transformed

# 統合された点群データに反射強度が含まれていることを確認
print("Shape of accumulated points:", accumulated_points.shape)

# for 202
accumulated_points_202 = transform_points(accumulated_points, world_cam_transformation_matrix_202_tr)
accumulated_points_202 = transform_points(accumulated_points_202, world_cam_transformation_matrix_202_rt)
accumulated_points_202 = transform_points(accumulated_points_202, vehicle_camera_transformation_matrix_202_tr)
accumulated_points_202 = transform_points(accumulated_points_202, vehicle_camera_transformation_matrix_202_rt)


# Open3Dでの可視化
#accumulated_pcd = o3d.geometry.PointCloud()
#accumulated_pcd.points = o3d.utility.Vector3dVector(accumulated_points[:, :3])
accumulated_pcd_202 = o3d.geometry.PointCloud()
accumulated_pcd_202.points = o3d.utility.Vector3dVector(accumulated_points_202[:, :3])

# 統合された点群を可視化
#o3d.visualization.draw_geometries([accumulated_pcd])

# 点群マップの保存
#o3d.io.write_point_cloud("3_flame_n_intensity_t.pcd", accumulated_pcd)
o3d.io.write_point_cloud("3_flame_n_intensity_t.pcd", accumulated_pcd_202)

########################################################################################################################反射強度の色付け

def apply_reflectance_to_colors(reflectance):
    """
    反射強度を基に色を設定する関数。反射強度はグレースケールの色に変換される。
    """
    # 反射強度を0から1の範囲に正規化
#    reflectance_normalized = (reflectance - np.min(reflectance)) / (np.max(reflectance) - np.min(reflectance))
    reflectance_normalized = (reflectance - np.min(reflectance)) / (0.7 * np.max(reflectance) - np.min(reflectance))
    colors = np.vstack([reflectance_normalized, reflectance_normalized, reflectance_normalized]).T
    return colors


# 反射強度を基に色を設定
reflectance_values = accumulated_points[:, 3]
colors = apply_reflectance_to_colors(reflectance_values)

# Open3Dでの可視化
#accumulated_pcd.colors = o3d.utility.Vector3dVector(colors)
accumulated_pcd_202.colors = o3d.utility.Vector3dVector(colors)
#o3d.visualization.draw_geometries([accumulated_pcd])

# 点群マップの保存
#o3d.io.write_point_cloud("3_flame_n_intensity_t_color.pcd", accumulated_pcd)
o3d.io.write_point_cloud("3_flame_n_intensity_t_color.pcd", accumulated_pcd_202)

############################################################################################カメラ座標系からスクリーン座標系に変換

#透視投影行列の作成
perspective_projection_matrix = np.zeros((3, 4))
perspective_projection_matrix[:3, :3] = camera_intrinsic_202


# Open3DのPointCloudから点の配列を取得
#points_3d = np.asarray(accumulated_pcd.points)
points_3d = np.asarray(accumulated_pcd_202.points)

# 3D点群データからカメラの後ろにある点を除外
print(points_3d.shape)
points_3d_filtered = points_3d[points_3d[:, 2] > 0]  # Z座標が0より大きい点のみを保持
print(points_3d_filtered.shape)
colors_filtered = colors[points_3d[:, 2] > 0]

# 同次座標系に変換
homogeneous_points_3d_filtered = np.hstack((points_3d_filtered, np.ones((points_3d_filtered.shape[0], 1))))

# 透視投影行列を適用して2D点に変換
projected_points_2d_filtered = perspective_projection_matrix @ homogeneous_points_3d_filtered.T

# 各点をそのw成分で割って正規化
projected_points_2d_filtered /= projected_points_2d_filtered[2, :]

# 正規化された点群を取得（最後の行を除去）
projected_points_2d_filtered = projected_points_2d_filtered[:2, :].T


# 2D点を画像に描画
#for point, color in zip(projected_points_2d_filtered, colors):
for point, color in zip(projected_points_2d_filtered, colors_filtered):
    x, y = int(point[0]), int(point[1])
    color_bgr = tuple(int(c * 255) for c in color[::-1])  # BGR形式に変換
    # 画像の範囲内にある点のみを描画する
    if 0 <= x < image_width_202 and 0 <= y < image_height_202:
        cv2.circle(camera_image_202, (x, y), 4, color_bgr, -1)

# 画像を表示
cv2.imshow('Camera Image with Projected LiDAR Points', camera_image_202)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 必要であれば画像をファイルに保存
cv2.imwrite('3flame_intensity_image.jpg', camera_image_202)

#nusc.render_pointcloud_in_image(my_sample_202['token'], pointsensor_channel='LIDAR_TOP')
