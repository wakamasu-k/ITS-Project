import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import matplotlib.pyplot as plt
from pyquaternion import Quaternion


# ========== 設定 ==========
NUSCENES_DIR = "nuscenes"  # nuScenesデータのパス
SAVE_DIR = "./output_reflectance"
scene_name = "scene-0061"  # シンガポール・昼間の例
CAMERA_CHANNEL = 'CAM_FRONT'
LIDAR_CHANNEL = 'LIDAR_TOP'

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== nuScenes 読み込み ==========
nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DIR, verbose=True)
scene = next(s for s in nusc.scene if s['name'] == scene_name)
first_sample = nusc.get('sample', scene['first_sample_token'])

# ========== サンプルからLiDARとCameraデータ取得 ==========
lidar_data = nusc.get('sample_data', first_sample['data'][LIDAR_CHANNEL])
camera_data = nusc.get('sample_data', first_sample['data'][CAMERA_CHANNEL])

# 画像サイズ取得
camera_image = cv2.imread(os.path.join(NUSCENES_DIR, camera_data['filename']))
img_h, img_w = camera_image.shape[:2]

# ========== キャリブレーション情報 ==========
lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
camera_calib = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])

# LiDAR → Ego
lidar_to_ego = np.eye(4)
lidar_to_ego[:3, :3] = Quaternion(lidar_calib['rotation']).rotation_matrix
lidar_to_ego[:3, 3] = np.array(lidar_calib['translation'])

# Ego → Global
ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
ego_to_global = np.eye(4)
ego_to_global[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
ego_to_global[:3, 3] = np.array(ego_pose['translation'])

# Camera Ego → Global（逆変換）
camera_ego_pose = nusc.get('ego_pose', camera_data['ego_pose_token'])
global_to_camera_ego = np.eye(4)
global_to_camera_ego[:3, :3] = Quaternion(camera_ego_pose['rotation']).rotation_matrix.T
global_to_camera_ego[:3, 3] = -global_to_camera_ego[:3, :3] @ np.array(camera_ego_pose['translation'])

camera_intrinsic = np.array(camera_calib['camera_intrinsic'])

# ========== 点群読み込み ==========
pc = LidarPointCloud.from_file(os.path.join(NUSCENES_DIR, lidar_data['filename']))
points = pc.points  # [x, y, z, intensity]

# ========== LiDAR → カメラ変換 ==========
# LiDAR -> ego -> global -> camera_ego -> camera_sensor
lidar_to_global = ego_to_global @ lidar_to_ego
points_hom = np.vstack((points[:3, :], np.ones(points.shape[1])))
points_global = np.dot(lidar_to_global, points_hom)
points_cam = np.dot(global_to_camera_ego, points_global)

# カメラ座標系におけるZ>0のみ
mask = points_cam[2, :] > 0
points_cam = points_cam[:, mask]
intensity = points[3, mask]

# ========== カメラ画像に投影 ==========
points_2d = view_points(points_cam[:3, :], camera_intrinsic, normalize=True)
mask = (points_2d[0, :] > 0) & (points_2d[0, :] < img_w) & (points_2d[1, :] > 0) & (points_2d[1, :] < img_h)
points_2d = points_2d[:, mask]
intensity = intensity[mask]

# ========== 反射強度画像を作成 ==========
reflectance_img = np.zeros((img_h, img_w), dtype=np.uint8)
intensity_norm = np.clip(255 * (intensity / np.max(intensity)), 0, 255).astype(np.uint8)

for i in range(points_2d.shape[1]):
    x, y = int(points_2d[0, i]), int(points_2d[1, i])
    cv2.circle(reflectance_img, (x, y), radius=6, color=int(intensity_norm[i]), thickness=-1)

# 低解像度化
reflectance_img_resized = cv2.resize(reflectance_img, (800, 450))

# ========== 保存と表示 ==========
save_path = os.path.join(SAVE_DIR, "reflectance_image.png")
cv2.imwrite(save_path, reflectance_img_resized)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Camera Image")
plt.imshow(cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Reflectance Image (RI)")
plt.imshow(reflectance_img_resized, cmap='gray')
plt.axis('off')
plt.show()

print(f"Reflectance image saved to: {save_path}")
