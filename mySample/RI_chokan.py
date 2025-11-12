import numpy as np
import cv2
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

# ====== データセットのパス ======
DATA_ROOT = "D:/Users/wakamatsu.k/Desktop/ITS/nuscenes"
nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)

# ====== 対象シーン ======
scene = nusc.scene[0]
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

# ====== 鳥瞰マップのパラメータ ======
MAP_SIZE = 200  # 地図範囲 [m]（200m四方）
RESOLUTION = 0.1  # 1ピクセル = 0.1m
IMG_SIZE = int(MAP_SIZE / RESOLUTION)

intensity_map = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
count_map = np.zeros_like(intensity_map)

# ====== 点群読み込みループ ======
while True:
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    pcl_path = f"{nusc.dataroot}/{lidar_data['filename']}"

    # LiDAR読み込み
    pc = LidarPointCloud.from_file(pcl_path)
    xyz = pc.points[:3, :]  # (x, y, z)
    intensity = pc.points[3, :]  # 反射強度

    # キャリブレーションと姿勢を取得
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # 座標変換（LiDAR→Ego→World）
    cs_trans = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']), inverse=False)
    pose_trans = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']), inverse=False)
    pc.transform(cs_trans)
    pc.transform(pose_trans)

    # 世界座標を取得
    points_world = pc.points.T[:, :3]
    intensity = pc.points.T[:, 3]

    # ====== 鳥瞰マップに投影 ======
    x, y = points_world[:, 0], points_world[:, 1]
    cx = np.median(x)  # 中心（車の経路中心に合わせる場合）
    cy = np.median(y)

    u = ((x - cx) / RESOLUTION + IMG_SIZE / 2).astype(np.int32)
    v = ((y - cy) / RESOLUTION + IMG_SIZE / 2).astype(np.int32)

    mask = (u >= 0) & (u < IMG_SIZE) & (v >= 0) & (v < IMG_SIZE)
    u, v, intensity = u[mask], v[mask], intensity[mask]

    for i in range(len(u)):
        intensity_map[v[i], u[i]] += intensity[i]
        count_map[v[i], u[i]] += 1

    # 次のフレームへ
    if sample['next'] == "":
        break
    else:
        sample = nusc.get('sample', sample['next'])

# ====== 平均強度を計算 ======
count_map[count_map == 0] = 1
reflectivity = intensity_map / count_map

# ====== 画像化 ======
reflectivity = np.clip(reflectivity, 0, np.percentile(reflectivity, 99))  # 外れ値除去
reflectivity_norm = cv2.normalize(reflectivity, None, 0, 255, cv2.NORM_MINMAX)
reflectivity_img = cv2.convertScaleAbs(reflectivity_norm)

# 上下反転（OpenCV座標系調整）
reflectivity_img = cv2.flip(reflectivity_img, 0)

cv2.imwrite("nuscenes_reflectivity_map.png", reflectivity_img)
cv2.imshow("Reflectivity Map", reflectivity_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ 鳥瞰反射強度マップを 'nuscenes_reflectivity_map.png' に保存しました。")
