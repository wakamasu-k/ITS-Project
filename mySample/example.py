import os
import numpy as np
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
import matplotlib.pyplot as plt

# =====================
# 設定
# =====================
NUSCENES_DIR = r"nuscenes"  # ← 自分のパスに変更
SAVE_DIR = r"mySample\output_reflectance_final"
scene_name = "scene-0061"
CAMERA_CHANNEL = "CAM_FRONT"
LIDAR_CHANNEL = "LIDAR_TOP"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# nuScenes データ読み込み
# =====================
nusc = NuScenes(version="v1.0-mini", dataroot=NUSCENES_DIR, verbose=True)
scene = next(s for s in nusc.scene if s["name"] == scene_name)
sample = nusc.get("sample", scene["first_sample_token"])

lidar_data = nusc.get("sample_data", sample["data"][LIDAR_CHANNEL])
camera_data = nusc.get("sample_data", sample["data"][CAMERA_CHANNEL])

# カメラ画像を読み込み
camera_image_path = os.path.join(NUSCENES_DIR, camera_data["filename"])
camera_image = cv2.imread(camera_image_path)
img_h, img_w = camera_image.shape[:2]  # 900×1600

# =====================
# キャリブレーション情報
# =====================
lidar_calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
camera_calib = nusc.get("calibrated_sensor", camera_data["calibrated_sensor_token"])

# LiDAR→Ego
lidar_to_ego = np.eye(4)
lidar_to_ego[:3, :3] = Quaternion(lidar_calib["rotation"]).rotation_matrix
lidar_to_ego[:3, 3] = np.array(lidar_calib["translation"])

# Ego→Global
ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
ego_to_global = np.eye(4)
ego_to_global[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
ego_to_global[:3, 3] = np.array(ego_pose["translation"])

# Global→CameraEgo
camera_ego_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])
global_to_camera_ego = np.eye(4)
global_to_camera_ego[:3, :3] = Quaternion(camera_ego_pose["rotation"]).rotation_matrix.T
global_to_camera_ego[:3, 3] = -global_to_camera_ego[:3, :3] @ np.array(camera_ego_pose["translation"])

# CameraEgo→CameraSensor
camera_sensor = np.eye(4)
camera_sensor[:3, :3] = Quaternion(camera_calib["rotation"]).rotation_matrix.T
camera_sensor[:3, 3] = -camera_sensor[:3, :3] @ np.array(camera_calib["translation"])

camera_intrinsic = np.array(camera_calib["camera_intrinsic"])

# =====================
# LiDAR点群読み込み
# =====================
lidar_path = os.path.join(NUSCENES_DIR, lidar_data["filename"])
pc = LidarPointCloud.from_file(lidar_path)
points = pc.points  # shape (4, N) → x, y, z, intensity

# =====================
# LiDAR → カメラ変換
# =====================
lidar_to_global = ego_to_global @ lidar_to_ego
points_hom = np.vstack((points[:3, :], np.ones(points.shape[1])))
points_global = lidar_to_global @ points_hom
points_cam = camera_sensor @ (global_to_camera_ego @ points_global)

# カメラ座標系でz>0のみ
mask = points_cam[2, :] > 0
points_cam = points_cam[:, mask]
intensity = points[3, mask]

# =====================
# 投影 & ソート
# =====================
# 奥から手前にソート
sorted_indices = np.argsort(points_cam[2, :])[::-1]
points_cam = points_cam[:, sorted_indices]
intensity = intensity[sorted_indices]

points_2d = view_points(points_cam[:3, :], camera_intrinsic, normalize=True)
mask = (points_2d[0, :] > 0) & (points_2d[0, :] < img_w) & (points_2d[1, :] > 0) & (points_2d[1, :] < img_h)
points_2d = points_2d[:, mask]
intensity = intensity[mask]
depth = points_cam[2, mask]

# =====================
# 距離補正 + 分位正規化 + ガンマ補正
# =====================
intensity_corrected = intensity / (np.sqrt(depth) + 1e-3)
max_val = np.percentile(intensity_corrected, 99)  # 上位1%をスケーリング上限に
intensity_norm = np.clip(255 * intensity_corrected / max_val, 0, 255).astype(np.uint8)

# =====================
# 反射強度画像生成
# =====================
reflectance_img = np.zeros((img_h, img_w), dtype=np.uint8)
for i in range(points_2d.shape[1]):
    x, y = int(points_2d[0, i]), int(points_2d[1, i])
    cv2.circle(reflectance_img, (x, y), radius=6, color=int(intensity_norm[i]), thickness=-1)

# =====================
# GaussianBlur + ガンマ補正
# =====================
reflectance_smooth = cv2.GaussianBlur(reflectance_img, (5, 5), 0)
reflectance_gamma = np.power(reflectance_smooth / 255.0, 0.6)
reflectance_gamma = (reflectance_gamma * 255).astype(np.uint8)

# =====================
# 低解像度化（800×450）
# =====================
reflectance_resized = cv2.resize(reflectance_gamma, (800, 450))
camera_resized = cv2.resize(camera_image, (800, 450))

# =====================
# 保存・表示
# =====================
cv2.imwrite(os.path.join(SAVE_DIR, "reflectance_image_final.png"), reflectance_resized)
cv2.imwrite(os.path.join(SAVE_DIR, "camera_resized.png"), camera_resized)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Camera Image (800x450)")
plt.imshow(cv2.cvtColor(camera_resized, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Final Reflectance Image (800x450)")
plt.imshow(reflectance_resized, cmap='gray')
plt.axis('off')

plt.show()
print("✅ Reflectance image saved:", os.path.join(SAVE_DIR, "reflectance_image_final.png"))
