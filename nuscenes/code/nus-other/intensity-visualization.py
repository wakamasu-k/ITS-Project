#一つのフレームの反射強度マッピング
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# .binファイルから点群データを読み込む
file_path = 'C:\\Users\\divin\\Downloads\\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448744447639.pcd.bin'
with open(file_path, 'rb') as f:
    content = f.read()
    lidar_data = np.frombuffer(content, dtype=np.float32).reshape(-1, 5)  # x, y, z, intensity, ring index

# XYZ座標と反射強度を抽出
xyz = lidar_data[:, :3]
intensities = lidar_data[:, 3]

# 反射強度を色にマッピング
intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
colors = plt.cm.jet(intensities)[:, :3]  # jet colormapを使用して色をマッピング

# 点群をOpen3DのPointCloudオブジェクトに変換し、色を設定
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 点群を可視化
o3d.visualization.draw_geometries([pcd])


# 保存先のファイル名
#save_filename = "map1_intensity.pcd"

# pcdオブジェクトを.pcd形式で保存
#o3d.io.write_point_cloud(save_filename, pcd)
#print(f"Point cloud saved to {save_filename}")


