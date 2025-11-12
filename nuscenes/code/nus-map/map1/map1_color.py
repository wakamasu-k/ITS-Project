#点群マップにz値で色を付ける
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt  # Added this line to define plt

def colorize_pointcloud(input_path, output_path):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    
    # Get the points from the point cloud
    points = np.asarray(pcd.points)
    
    # Get the z values (heights)
    z = points[:, 2]
    
    # Scale z values between 0 and 1
    scaled_z = (z - z.min()) / (z.max() - z.min())
    
    # Get colors using the jet colormap
    colors = plt.cm.jet(scaled_z)[:, :3]  # Added this line to get colors
    
    # Assign colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save the colored point cloud
    o3d.io.write_point_cloud(output_path, pcd)

if __name__ == "__main__":
    input_path = "C:\\Users\\divin\\Downloads\\scene1\\map1_LiDAR\\map1_intensity\\map1_intensity_no_clolor.pcd"  # Corrected the input path
    output_path = "C:\\Users\\divin\\Downloads\\colored_map1.pcd"  # Corrected the output path
    colorize_pointcloud(input_path, output_path)
