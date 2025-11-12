import numpy as np
import cv2
import os
from tqdm import tqdm

# =====================================================
# LiDAR点群 (.bin) → 反射強度画像(RI) 生成関数
# =====================================================
def generate_reflectance_image(lidar_bin_path, output_path, width=2048, height=32):
    # nuScenes の LIDAR_TOP.bin は (x, y, z, intensity, ring)
    pc = np.fromfile(lidar_bin_path, dtype=np.float32).reshape(-1, 5)
    xyz, intensity = pc[:, :3], pc[:, 3]

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # 角度を算出（水平: azimuth, 垂直: elevation）
    azimuth = np.arctan2(y, x)  # -π ～ π
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))  # -π/2 ～ π/2

    # 正規化して画像座標へ変換
    azimuth_norm = (azimuth - azimuth.min()) / (azimuth.max() - azimuth.min())
    elevation_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())

    u = np.clip((azimuth_norm * (width - 1)).astype(np.int32), 0, width - 1)
    v = np.clip(((1 - elevation_norm) * (height - 1)).astype(np.int32), 0, height - 1)

    # 反射強度を [0,255] にスケーリング
    intensity_scaled = np.clip(intensity / np.max(intensity) * 255, 0, 255).astype(np.uint8)

    # 同一画素に複数点が重なるため、z（距離）が小さい点を優先
    depth = np.sqrt(x**2 + y**2 + z**2)
    img_depth = np.full((height, width), np.inf)
    intensity_img = np.zeros((height, width), dtype=np.uint8)

    for i in range(len(u)):
        if depth[i] < img_depth[v[i], u[i]]:
            img_depth[v[i], u[i]] = depth[i]
            intensity_img[v[i], u[i]] = intensity_scaled[i]

    # 画像を保存
    cv2.imwrite(output_path, intensity_img)


# =====================================================
# sweeps/LIDAR_TOP の全ファイルに対して実行
# =====================================================
sweeps_folder = r"nuscenes\sweeps\LIDAR_TOP"
output_folder = r"nuscenes\intensity_sweeps"
os.makedirs(output_folder, exist_ok=True)

bin_files = [f for f in os.listdir(sweeps_folder) if f.endswith(".bin")]

for filename in tqdm(bin_files, desc="Generating Reflectance Images"):
    input_path = os.path.join(sweeps_folder, filename)
    output_path = os.path.join(output_folder, filename.replace(".bin", "_RI.png"))
    generate_reflectance_image(input_path, output_path)

print("✅ すべてのRI画像の生成が完了しました。")
