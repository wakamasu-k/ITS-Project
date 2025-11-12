import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("C:\\Users\\divin\\Downloads\\maps\\161-201.pcd")
points = np.asarray(pcd.points)
z_values = points[:, 2]
z_min, z_max = z_values.min(), z_values.max()

# ▼ (1) カラーマップを離散化する
num_levels = 20  # 色分けの段階数 (例: 20段階)
cmap_discrete = plt.get_cmap('jet', num_levels)

# ▼ (2) Z値を0～(num_levels-1)にマッピングする
norm_z = (z_values - z_min) / (z_max - z_min + 1e-8)
indices = np.floor(norm_z * (num_levels - 1)).astype(int)
# 0～(num_levels-1)の整数値に変換

# ▼ (3) 離散カラーマップを適用
discrete_colors = cmap_discrete(indices)

# ▼ (4) 点群の色を更新
pcd.colors = o3d.utility.Vector3dVector(discrete_colors[:, :3])

o3d.visualization.draw_geometries([pcd])



# 保存先のファイル名
save_filename = "color.pcd"

# pcdオブジェクトを.pcd形式で保存
o3d.io.write_point_cloud(save_filename, pcd)
print(f"Point cloud saved to {save_filename}")

#print(f"Point cloud saved to {save_filename}")

'''
#y

# Convert point cloud to numpy array
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Compute the min and max values for rescaling
min_val = np.min(points, axis=0)
max_val = np.max(points, axis=0)

# Normalize the y values between 0 and 1
normalized_y = (points[:, 1] - min_val[1]) / (max_val[1] - min_val[1])

# Apply stretching factor for the colormap
stretch_factor = 1 # This will make more points appear at the extremes of the colormap
stretched_y = np.clip(normalized_y * stretch_factor, 0, 1)

# Apply Jet colormap
jet_colors = plt.cm.jet(stretched_y)

# Update the colors in the original point cloud object
pcd.colors = o3d.utility.Vector3dVector(jet_colors[:, :3])

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# Save the point cloud with colors
save_filename = "3_flame_n_intensity_t_y.pcd"
o3d.io.write_point_cloud(save_filename, pcd)
print(f"Point cloud saved to {save_filename}")

#x

# Convert point cloud to numpy array
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Compute the min and max values for rescaling
min_val = np.min(points, axis=0)
max_val = np.max(points, axis=0)

# Normalize the x values between 0 and 1
normalized_x = (points[:, 0] - min_val[0]) / (max_val[0] - min_val[0])

# Apply stretching factor for the colormap
stretch_factor = 1 # Adjust the range of colors
stretched_x = np.clip(normalized_x * stretch_factor, 0, 1)

# Apply Jet colormap
jet_colors = plt.cm.jet(stretched_x)

# Update the colors in the original point cloud object
pcd.colors = o3d.utility.Vector3dVector(jet_colors[:, :3])

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# Save the point cloud with colors
save_filename = "3_flame_n_intensity_t_x.pcd"
o3d.io.write_point_cloud(save_filename, pcd)
print(f"Point cloud saved to {save_filename}")

'''