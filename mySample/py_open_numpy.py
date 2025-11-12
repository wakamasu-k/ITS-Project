import os
import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

# === 設定 ===
NUSCENES_ROOT = r"nuscenes"
OVERLAY_SAVE_DIR = os.path.join(NUSCENES_ROOT, "overlays")
os.makedirs(OVERLAY_SAVE_DIR, exist_ok=True)

# 利用可能なカメラ
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# nuScenes の初期化
nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_ROOT, verbose=False)

# === LiDAR と画像を重ね合わせる関数 ===
def project_lidar_to_image(lidar_sd, cam_sd):
    """LiDAR 点群をカメラ画像に投影して合成画像を返す"""

    # 1. パラメータ取得
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    cam_ego = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    # 2. LiDAR→Camera 変換行列（ego_poseを挟んで正確に計算）
    T_lidar_sensor_to_ego = transform_matrix(
        np.array(lidar_cs['translation']),
        Quaternion(lidar_cs['rotation']),
        inverse=False
    )
    T_ego_lidar_to_global = transform_matrix(
        np.array(lidar_ego['translation']),
        Quaternion(lidar_ego['rotation']),
        inverse=False
    )
    T_global_to_ego_cam = transform_matrix(
        np.array(cam_ego['translation']),
        Quaternion(cam_ego['rotation']),
        inverse=True
    )
    T_ego_cam_to_camera = transform_matrix(
        np.array(cam_cs['translation']),
        Quaternion(cam_cs['rotation']),
        inverse=True
    )
    T_lidar_to_cam = T_ego_cam_to_camera @ T_global_to_ego_cam @ T_ego_lidar_to_global @ T_lidar_sensor_to_ego

    # 3. カメラ内部パラメータ
    K = np.array(cam_cs['camera_intrinsic'])

    # 4. LiDAR点群を読み込み
    lidar_path = os.path.join(NUSCENES_ROOT, lidar_sd['filename'])
    if lidar_path.endswith('.bin'):
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    elif lidar_path.endswith('.pcd'):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(lidar_path)
        points = np.asarray(pcd.points)
    else:
        raise ValueError("Unsupported LiDAR format")

    # 5. 齢換適用
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam = (T_lidar_to_cam @ points_hom.T).T[:, :3]

    # Z>0のみ（カメラ前方）
    points_cam = points_cam[points_cam[:, 2] > 0]

    # 6. 投影
    uv = (K @ points_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    # 7. カメラ画像を読み込み
    img_path = os.path.join(NUSCENES_ROOT, cam_sd['filename'])
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # 8. 範囲内の点のみ残す
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[mask]
    depths = points_cam[mask, 2]

    # 9. 距離に応じて色をつける（近:赤 → 遠:青）
    depths_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)
    colors = cv2.applyColorMap((255 - (depths_norm * 255)).astype(np.uint8), cv2.COLORMAP_JET)

    for (u, v), c in zip(uv.astype(int), colors):
        cv2.circle(image, (u, v), 1, tuple(int(x) for x in c[0]), -1)

    return image

# === メイン処理 ===
def main():
    # LiDARデータ一覧
    lidar_sweeps = [s for s in nusc.sample_data if 'LIDAR_TOP' in s['channel']]
    lidar_sweeps.sort(key=lambda x: x['timestamp'])

    print(f"利用可能な LiDAR sweeps 数: {len(lidar_sweeps)}")
    idx = int(input(f"表示したい sweep の番号を 0〜{len(lidar_sweeps)-1} で入力してください: "))

    lidar_sd = lidar_sweeps[idx]

    for cam_name in CAMERAS:
        cam_sweeps = [s for s in nusc.sample_data if cam_name in s['channel']]
        cam_sweeps.sort(key=lambda x: x['timestamp'])
        cam_sd = cam_sweeps[idx]

        print(f"▶ {cam_name}: {cam_sd['filename']}")
        overlay = project_lidar_to_image(lidar_sd, cam_sd)

        save_path = os.path.join(OVERLAY_SAVE_DIR, f"overlay_{cam_name}_{idx}.jpg")
        cv2.imwrite(save_path, overlay)
        cv2.imshow(cam_name, overlay)

    print(f"\n✅ すべてのカメラ画像を {OVERLAY_SAVE_DIR} に保存しました。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
