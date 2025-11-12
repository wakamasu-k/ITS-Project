# nuscenes_reflectance_image.py
import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm

# ===========================
# 設定
# ===========================
DATA_ROOT = r"nuscenes"
NU_VERSION = "v1.0-mini"
OUTPUT_DIR = "output/reflectance_image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LiDARセンサパラメータ（nuScenesは64層LiDAR）
IMG_W = 2048   # 水平方向解像度（角度分解能）
IMG_H = 64     # 垂直方向解像度（レーザ層数）

# ===========================
# nuScenes ロード
# ===========================
nusc = NuScenes(version=NU_VERSION, dataroot=DATA_ROOT, verbose=False)

for scene in tqdm(nusc.scene, desc="Processing scenes"):
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    while True:
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])

        # 点群読み込み
        pc = LidarPointCloud.from_file(lidar_path)
        pts = pc.points.T  # shape = (N, 5): x, y, z, intensity, ring
        x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

        # === LiDAR視点で投影 ===
        r = np.sqrt(x**2 + y**2 + z**2)
        yaw = np.arctan2(y, x)               # 水平角 [-π, π]
        pitch = np.arcsin(z / r)             # 垂直角 [-π/2, π/2]

        # ピクセル座標に変換
        u = (yaw + np.pi) / (2 * np.pi) * IMG_W
        v = (1 - (pitch + np.pi/4) / (np.pi/2)) * IMG_H
        u = np.clip(u.astype(np.int32), 0, IMG_W - 1)
        v = np.clip(v.astype(np.int32), 0, IMG_H - 1)

        # === 反射強度マップ生成 ===
        img = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        for i in range(len(u)):
            if intensity[i] > img[v[i], u[i]]:
                img[v[i], u[i]] = intensity[i]

        # 正規化
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        # 保存
        scene_name = scene['name'].replace("/", "_")
        fname = f"{scene_name}_{lidar_data['timestamp']}_reflectance.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img)

        # 次のサンプルへ
        if sample['next'] == "":
            break
        sample = nusc.get('sample', sample['next'])
