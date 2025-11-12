# nuscenes_mini_reflectivity.py
import os
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

# ----------------------------
# 設定（環境に合わせて変更）
# ----------------------------
DATA_ROOT = r"D:/Users/wakamatsu.k/Desktop/ITS/nuscenes"  # nuscenes データルート（v1.0-mini を含むフォルダ）
NU_VERSION = "v1.0-mini"
OUT_DIR = "output/reflectivity_mini"   # 出力フォルダ
MAP_SIZE = 150.0       # 地図サイズ [m]（1辺） — シーンに合わせて調整
RESOLUTION = 0.1       # m / pixel（解像度）
PERCENTILE = 99.0      # 外れ値カットのパーセンタイル
VOXEL_DOWN_SAMPLE = None  # (例: 0.05) を入れると点群を先にボクセルダウンサンプル（m）
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading NuScenes (this may take a few seconds)...")
nusc = NuScenes(version=NU_VERSION, dataroot=DATA_ROOT, verbose=False)

IMG_SIZE = int(MAP_SIZE / RESOLUTION)
print(f"Map params: MAP_SIZE={MAP_SIZE}m, RES={RESOLUTION}m -> IMG_SIZE={IMG_SIZE}px")

def process_scene(scene_record):
    """1シーン分のRIマップを作成して返す（配列）と、シーンの中心 (cx,cy) と軌跡座標を返す"""
    first_token = scene_record['first_sample_token']
    sample = nusc.get('sample', first_token)

    # 累積用配列
    intensity_sum = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    count = np.zeros_like(intensity_sum)

    # 軌跡（egoの位置）を保存しておく（描画用）
    traj_x = []
    traj_y = []

    # シーン内の全点を集めずにフレーム毎に投影して加算する方法（メモリ対策）
    # ただし、中心(cx,cy) を決めるために最初にシーン全体の中央値を求めるための一時保存を使う。
    coords_for_center = []

    # --- 1回目ループ：中心決定のために座標のサンプルを得る ---
    tmp_sample = sample
    while True:
        lidar_token = tmp_sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        pcl_path = os.path.join(nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        # キャリブ & ego pose
        cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # 変換（Quaternionが必要）
        pc.transform(transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False))
        pc.transform(transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=False))

        pts_world = pc.points[:3, :].T  # (N,3)
        coords_for_center.append(pts_world[np.random.choice(pts_world.shape[0], min(2000, pts_world.shape[0]), replace=False)])

        # 軌跡記録（車体位置 = pose['translation']）
        traj_x.append(pose['translation'][0])
        traj_y.append(pose['translation'][1])

        if tmp_sample['next'] == "":
            break
        tmp_sample = nusc.get('sample', tmp_sample['next'])

    coords_for_center = np.concatenate(coords_for_center, axis=0)
    cx = float(np.median(coords_for_center[:, 0]))
    cy = float(np.median(coords_for_center[:, 1]))

    # --- 2回目ループ：地図への投影と強度集計 ---
    sample = nusc.get('sample', first_token)
    pbar = tqdm(desc=f"Processing scene {scene_record['name']}", unit='frame')
    while True:
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        pcl_path = os.path.join(nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        # キャリブ & ego pose
        cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        pc.transform(transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False))
        pc.transform(transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=False))

        if VOXEL_DOWN_SAMPLE:
            # simple grid downsample by integer binning on x,y,z
            pts = pc.points.T
            # naive downsample: quantize coordinates
            keys = (np.floor(pts[:, :3] / VOXEL_DOWN_SAMPLE)).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            pts = pts[idx]
        else:
            pts = pc.points.T  # (N, >=4) x,y,z,intensity,...

        if pts.shape[1] < 4:
            pbar.update(1)
            if sample['next'] == "":
                break
            sample = nusc.get('sample', sample['next'])
            continue

        x = pts[:, 0]
        y = pts[:, 1]
        intensity = pts[:, 3]

        # ピクセル座標
        u = np.floor((x - cx) / RESOLUTION + IMG_SIZE / 2).astype(np.int32)
        v = np.floor((y - cy) / RESOLUTION + IMG_SIZE / 2).astype(np.int32)

        mask = (u >= 0) & (u < IMG_SIZE) & (v >= 0) & (v < IMG_SIZE)
        u = u[mask]
        v = v[mask]
        intensity = intensity[mask]

        # 加算
        intensity_sum[v, u] += intensity
        count[v, u] += 1

        pbar.update(1)
        if sample['next'] == "":
            break
        sample = nusc.get('sample', sample['next'])
    pbar.close()

    # 平均を計算
    with np.errstate(divide='ignore', invalid='ignore'):
        count_nonzero = count.copy()
        count_nonzero[count_nonzero == 0] = 1
        reflectivity = intensity_sum / count_nonzero
        reflectivity[count == 0] = 0.0

    return reflectivity, (cx, cy), (traj_x, traj_y)

# ---------- メイン処理 ----------
scene_list = nusc.scene
print(f"Found {len(scene_list)} scenes in {NU_VERSION}")

for scene_rec in scene_list:
    reflectivity, center, traj = process_scene(scene_rec)

    # クリッピングと正規化
    vmax = np.percentile(reflectivity[reflectivity > 0], PERCENTILE) if np.any(reflectivity > 0) else 1.0
    reflectivity_clipped = np.clip(reflectivity, 0, vmax)
    # 0-255 にスケール
    reflectivity_norm = (reflectivity_clipped / vmax * 255.0).astype(np.uint8)

    # 上下反転しておく（y軸方向の表示調整）
    img = np.flipud(reflectivity_norm)

    # 保存（メイン画像）
    scene_name = scene_rec['name'].replace("/", "_")
    png_path = os.path.join(OUT_DIR, f"{scene_name}_RI.png")
    plt.imsave(png_path, img, cmap='gray', vmin=0, vmax=255)
    print(f"Saved: {png_path}")

    # 軌跡を重ねた画像（オプション）
    traj_x, traj_y = traj
    if len(traj_x) > 0:
        # 軌跡をピクセル座標に変換（同じ cx,cy と RESOLUTION を使用）
        cx, cy = center
        tx = ((np.array(traj_x) - cx) / RESOLUTION + IMG_SIZE / 2)
        ty = ((np.array(traj_y) - cy) / RESOLUTION + IMG_SIZE / 2)
        # flip y to match image
        ty = IMG_SIZE - ty

        # 描画して保存
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.plot(tx, ty, color='red', linewidth=1)
        ax.set_axis_off()
        overlay_path = os.path.join(OUT_DIR, f"{scene_name}_RI_traj.png")
        fig.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved trajectory overlay: {overlay_path}")

print("All scenes processed.")
