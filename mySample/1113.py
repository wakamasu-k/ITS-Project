import os
import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from tqdm import tqdm

# === 設定 ===
NUSCENES_ROOT = r"nuscenes"
VERSION = "v1.0-mini"
OUTPUT_DIR = os.path.join(NUSCENES_ROOT, "overlays_highprec")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# nuScenes 初期化
nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)


def project_lidar_to_camera(lidar_points, lidar_sd, cam_sd):
    """
    LiDAR点群をカメラ座標系に変換し、画像上に投影
    """

    # --- Calibration / Pose 取得 ---
    lidar_calib = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    cam_calib = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    # --- LiDAR -> global ---
    T_lidar_to_ego = transform_matrix(
        np.array(lidar_calib["translation"]),
        Quaternion(lidar_calib["rotation"]),
        inverse=False,
    )
    T_ego_to_global = transform_matrix(
        np.array(lidar_pose["translation"]),
        Quaternion(lidar_pose["rotation"]),
        inverse=False,
    )

    # --- global -> camera ---
    T_global_to_ego_cam = transform_matrix(
        np.array(cam_pose["translation"]),
        Quaternion(cam_pose["rotation"]),
        inverse=True,
    )
    T_ego_cam_to_cam = transform_matrix(
        np.array(cam_calib["translation"]),
        Quaternion(cam_calib["rotation"]),
        inverse=True,
    )

    # 合成行列
    T_lidar_to_cam = T_ego_cam_to_cam @ T_global_to_ego_cam @ T_ego_to_global @ T_lidar_to_ego

    # 点群をカメラ座標系へ
    points_h = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))))
    points_cam = (T_lidar_to_cam @ points_h.T).T[:, :3]

    # カメラ前方のみ残す
    points_cam = points_cam[points_cam[:, 2] > 0]

    # --- 画像投影 ---
    K = np.array(cam_calib["camera_intrinsic"])
    uv = (K @ points_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    return uv, points_cam


def draw_projection(image, uv, points_cam):
    """点群を画像に描画（深度カラーマップ）"""
    h, w = image.shape[:2]
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[mask]
    depths = points_cam[mask, 2]

    depths_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)
    colors = cv2.applyColorMap(
        (255 - (depths_norm * 255)).astype(np.uint8), cv2.COLORMAP_JET
    )

    for (u, v), color in zip(uv.astype(int), colors):
        cv2.circle(image, (u, v), 2, tuple(int(c) for c in color[0]), -1)

    return image


def main():
    # 任意のサンプル選択
    scene = nusc.scene[0]
    first_sample_token = scene["first_sample_token"]
    sample = nusc.get("sample", first_sample_token)

    while True:
        # LiDAR データ取得
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(NUSCENES_ROOT, lidar_sd["filename"])
        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

        # 各カメラごとに処理
        for cam_name in CAMERAS:
            cam_token = sample["data"][cam_name]
            cam_sd = nusc.get("sample_data", cam_token)
            img_path = os.path.join(NUSCENES_ROOT, cam_sd["filename"])

            image = cv2.imread(img_path)
            uv, pts_cam = project_lidar_to_camera(lidar_points, lidar_sd, cam_sd)
            overlay = draw_projection(image, uv, pts_cam)

            save_path = os.path.join(OUTPUT_DIR, f"{cam_name}_{lidar_sd['token']}.jpg")
            cv2.imwrite(save_path, overlay)
            cv2.imshow(cam_name, overlay)

        print(f"✅ 保存完了 → {OUTPUT_DIR}")

        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif sample["next"] == "":
            break
        else:
            sample = nusc.get("sample", sample["next"])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
