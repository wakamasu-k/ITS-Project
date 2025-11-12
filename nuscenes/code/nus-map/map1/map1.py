###点群マップを作る

from nuscenes.nuscenes import NuScenes
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion

# Load dataset
nusc = NuScenes(version='v1.0-mini', dataroot='C:\\Users\\divin\\OneDrive - 梅村学園　中京大学\\MD\\卒業研究\\data\\nuScenes mini us', verbose=True)

# Helper function to create a transformation matrix from translation and rotation
def transform_to_matrix(translation, rotation):
    q = Quaternion(rotation)
    matrix = np.eye(4)
    matrix[:3, :3] = q.rotation_matrix
    matrix[:3, 3] = translation
    return matrix

# Get calibration data
calibration_data = nusc.calibrated_sensor[17]
calibration_transform = transform_to_matrix(calibration_data['translation'], calibration_data['rotation'])

# File paths and corresponding poses
file_paths = [
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604547893.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605047769.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605548192.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151606048630.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151606549066.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151607048933.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151607548824.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151608048151.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151608548020.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151609047890.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151609547766.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151609947025.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151610446899.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151610946785.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151611446636.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151611896734.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151612397179.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151612897588.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151613398020.pcd.bin",#後半
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151613948222.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151614450164.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151614948536.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151615448397.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151615947733.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151616447606.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151616947490.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151617397050.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151617947237.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151618447134.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151618946993.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151619446866.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151619947313.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151620447745.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151621047706.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151621448049.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151621947928.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151622448916.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene1\\LiDAR\\lidar_scene1\\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151622948227.pcd.bin",
]
poses = [nusc.ego_pose[4238], nusc.ego_pose[4282], nusc.ego_pose[4293], nusc.ego_pose[4258], 
         nusc.ego_pose[4277], nusc.ego_pose[4322], nusc.ego_pose[4333], nusc.ego_pose[4297],
         nusc.ego_pose[4316], nusc.ego_pose[4362], nusc.ego_pose[4372], nusc.ego_pose[4336],
         nusc.ego_pose[4356], nusc.ego_pose[4364], nusc.ego_pose[4409], nusc.ego_pose[4419],
         nusc.ego_pose[4382], nusc.ego_pose[4440], nusc.ego_pose[4413], nusc.ego_pose[4445],
         nusc.ego_pose[4455], nusc.ego_pose[4441], nusc.ego_pose[4490], nusc.ego_pose[4503],#後半
         nusc.ego_pose[4660], nusc.ego_pose[4480], nusc.ego_pose[4532], nusc.ego_pose[4545],
         nusc.ego_pose[4534], nusc.ego_pose[4520], nusc.ego_pose[4573], nusc.ego_pose[4586],
         nusc.ego_pose[4535], nusc.ego_pose[4560], nusc.ego_pose[4614], nusc.ego_pose[4564],
         nusc.ego_pose[4572], nusc.ego_pose[4598], nusc.ego_pose[4656], nusc.ego_pose[4666]
]



# Apply transformations and accumulate point clouds
accumulated_pcd = o3d.geometry.PointCloud()
for file_path, pose in zip(file_paths, poses):
    # Load point cloud data
    pcd_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data)
    
    # Apply calibration transformation
    pcd.transform(calibration_transform)
    
    # Apply ego pose transformation
    ego_transform = transform_to_matrix(pose['translation'], pose['rotation'])
    pcd.transform(ego_transform)
    
    # Merge with accumulated point cloud
    accumulated_pcd += pcd

# Visualize the merged point cloud
o3d.visualization.draw_geometries([accumulated_pcd])
# 非推奨 open3d.visualization.draw_geometries


# 点群マップの保存
o3d.io.write_point_cloud("map2.pcd", accumulated_pcd)



# 保存先のファイル名
#save_filename = "map1.pcd"

# pcdオブジェクトを.pcd形式で保存
#o3d.io.write_point_cloud(save_filename, accumulated_pcd)
#print(f"Point cloud saved to {save_filename}")

#print(f"Point cloud saved to {save_filename}")
