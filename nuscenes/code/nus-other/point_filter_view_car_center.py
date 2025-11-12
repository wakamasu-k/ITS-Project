#xyz
'''import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# .binファイルから点群データを読み込む
file_path = 'C:\\Users\\divin\\Downloads\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385092150099.pcd.bin'
with open(file_path, 'rb') as f:
    content = f.read()
    lidar_data = np.frombuffer(content, dtype=np.float32).reshape(-1, 5)  # x, y, z, intensity, ring index

# XYZ座標と反射強度を抽出
xyz = lidar_data[:, :3]
intensities = lidar_data[:, 3]

# 自車の中心からの距離を計算（原点からの距離）
distances = np.linalg.norm(xyz, axis=1)
threshold_distance = 2  # 1メートルの閾値

# 1メートル以上離れている点のみを保持
mask = distances > threshold_distance
xyz_filtered = xyz[mask]
intensities_filtered = intensities[mask]

# 反射強度を色にマッピング
intensities_filtered_normalized = (intensities_filtered - intensities_filtered.min()) / (intensities_filtered.max() - intensities_filtered.min())
colors = plt.cm.jet(intensities_filtered_normalized)[:, :3]

# 点群をOpen3DのPointCloudオブジェクトに変換し、色を設定
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 点群を可視化
o3d.visualization.draw_geometries([pcd])'''

#xy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# .binファイルから点群データを読み込む
file_path = 'C:\\Users\\divin\\Downloads\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385092150099.pcd.bin'
with open(file_path, 'rb') as f:
    content = f.read()
    lidar_data = np.frombuffer(content, dtype=np.float32).reshape(-1, 5)  # x, y, z, intensity, ring index

# XYZ座標と反射強度を抽出
xyz = lidar_data[:, :3]
intensities = lidar_data[:, 3]

# 自車の中心からXY平面上の距離を計算
distances_xy = np.linalg.norm(xyz[:, :2], axis=1)  # Z軸を無視
threshold_distance = 2.0  # 1メートルの閾値

# 1メートル以上離れている点のみを保持
mask = distances_xy > threshold_distance
xyz_filtered = xyz[mask]
intensities_filtered = intensities[mask]

# 反射強度を色にマッピング
intensities_filtered_normalized = (intensities_filtered - intensities_filtered.min()) / (intensities_filtered.max() - intensities_filtered.min())
colors = plt.cm.jet(intensities_filtered_normalized)[:, :3]

# 点群をOpen3DのPointCloudオブジェクトに変換し、色を設定
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 点群を可視化
o3d.visualization.draw_geometries([pcd])

