import cv2
import numpy as np
import os
import glob
import open3d as o3d
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes import NuScenes
from pyquaternion import Quaternion

# === nuScenes データセットのルート ===
dataset_root = r"D:\Users\wakamatsu.k\Desktop\ITS\nuscenes"

# === 各センサのパス ===
camera_dirs = {
    'CAM_FRONT': os.path.join(dataset_root, "sweeps", "CAM_FRONT"),
    'CAM_FRONT_LEFT': os.path.join(dataset_root, "sweeps", "CAM_FRONT_LEFT"),
    'CAM_FRONT_RIGHT': os.path.join(dataset_root, "sweeps", "CAM_FRONT_RIGHT"),
    'CAM_BACK': os.path.join(dataset_root, "sweeps", "CAM_BACK"),
    'CAM_BACK_LEFT': os.path.join(dataset_root, "sweeps", "CAM_BACK_LEFT"),
    'CAM_BACK_RIGHT': os.path.join(dataset_root, "sweeps", "CAM_BACK_RIGHT"),
}
lidar_dir = os.path.join(dataset_root, "sweeps", "LIDAR_TOP")

# === 出力フォルダ ===
output_dir = os.path.join(dataset_root, "overlay_results")
os.makedirs(output_dir, exist_ok=True)

# === 画像ファイルの探索 ===
image_lists = {}
for cam, path in camera_dirs.items():
    files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        files.extend(glob.glob(os.path.join(path, ext)))
    files = sorted(files)
    image_lists[cam] = files
    print(f"{cam}: {len(files)} 枚の画像を読み込みました ({path})")

# === LiDARファイル ===
lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.bin")))
print(f"LIDAR_TOP: {len(lidar_files)} 点群ファイルを読み込みました")

# === インデックスを入力 ===
index = int(input(f"\n表示したいフレーム番号を入力してください (0〜{len(lidar_files)-1}): "))

# === nuScenes のインスタンスを準備（キャリブ用） ===
nusc = NuScenes(version='v1.0-mini', dataroot=dataset_root, verbose=False)

# === 歪み補正と重ね合わせの処理 ===
for cam, img_list in image_lists.items():
    if len(img_list) <= index:
        print(f"{cam}: {index} 番目の画像は存在しません")
        continue

    img_path = img_list[index]
    lidar_path = lidar_files[index]

    print(f"\n=== {cam} を処理中 ===")
    print(f"Camera: {img_path}")
    print(f"Lidar : {lidar_path}")

    # === 画像を読み込み ===
    img_raw = cv2.imread(img_path)
    if img_raw is None:
        print(f"{cam}: 画像を読み込めませんでした")
        continue

    # === 歪み補正用パラメータ（例: nuScenes Miniデータのキャリブを想定） ===
    # 実際には calibration カタログから取得するのがベスト
    K = np.array([[1200, 0, img_raw.shape[1]/2],
                  [0, 1200, img_raw.shape[0]/2],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.15, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)

    # === 歪み補正 ===
    img_undistorted = cv2.undistort(img_raw, K, dist_coeffs)

    # === LiDAR点群読み込み ===
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # === 仮の外部パラメータ（LiDAR→Camera）===
    R = np.array([[0, -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]], dtype=np.float32)
    t = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    Rt = np.hstack((R, t))
    proj = K @ Rt

    # === 射影 ===
    pts_2d = proj @ points_hom.T
    pts_2d[:2] /= pts_2d[2]
    u, v = pts_2d[0], pts_2d[1]
    mask = (u >= 0) & (u < img_raw.shape[1]) & (v >= 0) & (v < img_raw.shape[0]) & (pts_2d[2] > 0)

    u, v = u[mask].astype(np.int32), v[mask].astype(np.int32)
    depth = pts_2d[2][mask]

    # === 色付け ===
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    colors = (cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET))
    colors = colors[:, 0, :]

    # === 重ね合わせ画像作成 ===
    img_overlay_raw = img_raw.copy()
    img_overlay_undist = img_undistorted.copy()

    for i in range(len(u)):
        c = tuple(int(vv) for vv in colors[i])
        cv2.circle(img_overlay_raw, (u[i], v[i]), 2, c, -1)
        cv2.circle(img_overlay_undist, (u[i], v[i]), 2, c, -1)

    # === 結果をまとめて表示 ===
    h, w = img_raw.shape[:2]
    combined = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    combined[0:h, 0:w] = img_raw
    combined[0:h, w:2*w] = img_undistorted
    combined[h:2*h, 0:w] = img_overlay_raw
    combined[h:2*h, w:2*w] = img_overlay_undist

    label_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (30, 50), label_font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Undistorted", (w+30, 50), label_font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Overlay (Raw)", (30, h+50), label_font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Overlay (Undistorted)", (w+30, h+50), label_font, 1, (255, 255, 255), 2)

    # === 表示 ===
    cv2.imshow(cam, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # === 保存 ===
    save_path = os.path.join(output_dir, f"{cam}_overlay_frame{index}.jpg")
    cv2.imwrite(save_path, combined)
    print(f"{cam}: {save_path} に保存しました ✅")
