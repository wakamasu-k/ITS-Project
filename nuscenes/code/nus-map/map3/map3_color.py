#z値の色付けを変える
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load the point cloud
pcd = o3d.io.read_point_cloud("C://Users//divin//Downloads//scene3_sgp//map_lidar/map3.pcd")

# Convert point cloud to numpy array
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Compute the min and max values for rescaling
min_val = np.min(points, axis=0)
max_val = np.max(points, axis=0)

# Normalize the points between 0 and 1
normalized_points = (points - min_val) / (max_val - min_val)

# Apply stretching factor for the colormap
stretch_factor = 1.0 # This will make more points appear as 'red'
stretched_z = np.clip(normalized_points[:, 2] * stretch_factor, 0, 1)

# Apply Jet colormap
jet_colors = plt.cm.jet(stretched_z)

# Update the colors in the original point cloud object
pcd.colors = o3d.utility.Vector3dVector(jet_colors[:, :3])

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# 保存先のファイル名
save_filename = "map3_color10.pcd"

# pcdオブジェクトを.pcd形式で保存
o3d.io.write_point_cloud(save_filename, pcd)
print(f"Point cloud saved to {save_filename}")

print(f"Point cloud saved to {save_filename}")
