import numpy as np
import cv2
import os

lidar_folder = r"D:\Users\wakamatsu.k\Desktop\ITS\nuscenes\lidar"
save_folder = r"D:\Users\wakamatsu.k\Desktop\ITS\nuscenes\intensity_bev"
os.makedirs(save_folder, exist_ok=True)

X_MIN, X_MAX = -50, 50
Y_MIN, Y_MAX = -50, 50
RESOLUTION = 0.1
WIDTH = int((Y_MAX - Y_MIN) / RESOLUTION)
HEIGHT = int((X_MAX - X_MIN) / RESOLUTION)

def generate_bev_intensity(pcd_path, save_path):
    points = np.fromfile(pcd_path, dtype=np.float32)

    # 点群フォーマット自動判定
    if len(points) % 5 == 0:
        points = points.reshape(-1, 5)
        x, y, z, intensity, ring = points.T
    elif len(points) % 4 == 0:
        points = points.reshape(-1, 4)
        x, y, z, intensity = points.T
    else:
        raise ValueError(f"Unexpected point data format: {pcd_path}")

    mask = (x > X_MIN) & (x < X_MAX) & (y > Y_MIN) & (y < Y_MAX)
    x, y, intensity = x[mask], y[mask], intensity[mask]

    x_img = ((x - X_MIN) / (X_MAX - X_MIN) * HEIGHT).astype(np.int32)
    y_img = ((y - Y_MIN) / (Y_MAX - Y_MIN) * WIDTH).astype(np.int32)

    x_img = np.clip(x_img, 0, HEIGHT - 1)
    y_img = np.clip(y_img, 0, WIDTH - 1)

    bev_image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    for xi, yi, inten in zip(x_img, y_img, intensity):
        bev_image[xi, yi] = max(bev_image[xi, yi], inten)

    bev_image = np.flipud(bev_image)
    bev_image_uint8 = (bev_image * 255).astype(np.uint8)
    cv2.imwrite(save_path, bev_image_uint8)
    print(f"✅ Saved BEV Intensity Image → {save_path}")

for filename in os.listdir(lidar_folder):
    if filename.endswith(".bin"):
        input_path = os.path.join(lidar_folder, filename)
        output_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}_RI.png")
        generate_bev_intensity(input_path, output_path)
