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
calibration_data = nusc.calibrated_sensor[77]
calibration_transform = transform_to_matrix(calibration_data['translation'], calibration_data['rotation'])

# File paths and corresponding poses
file_paths = [
   
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984233547259.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984234047134.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984234547551.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984234947904.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984235447825.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984235947081.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984236446970.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984236946289.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984237446737.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984237947155.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984238447571.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984238948012.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984239448435.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984239948316.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984240448191.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984240947518.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984241447959.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984241947812.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984242447730.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984242947023.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984243446332.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984243946219.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984244446640.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984244946513.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984245447134.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984245947391.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984246447815.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984246948247.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984247449248.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984247949081.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984248448966.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984248948287.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984249448168.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984249948039.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984250447364.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984250947236.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984251447166.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984251947010.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984252446890.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984252947312.pcd.bin",
"C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984253447765.pcd.bin",

]

poses = [

nusc.ego_pose[19965],
nusc.ego_pose[20035],
nusc.ego_pose[19995],
nusc.ego_pose[20003],
nusc.ego_pose[20017],
nusc.ego_pose[20012],
nusc.ego_pose[20086],
nusc.ego_pose[20042],
nusc.ego_pose[20057],
nusc.ego_pose[20052],
nusc.ego_pose[20126],
nusc.ego_pose[20082],
nusc.ego_pose[20097],
nusc.ego_pose[20092],
nusc.ego_pose[20167],
nusc.ego_pose[20123],
nusc.ego_pose[20138],
nusc.ego_pose[20132],
nusc.ego_pose[20211],
nusc.ego_pose[20164],
nusc.ego_pose[20176],
nusc.ego_pose[20170],
nusc.ego_pose[20252],
nusc.ego_pose[20205],
nusc.ego_pose[20214],
nusc.ego_pose[20208],
nusc.ego_pose[20293],
nusc.ego_pose[20246],
nusc.ego_pose[20255],
nusc.ego_pose[20248],
nusc.ego_pose[20337],
nusc.ego_pose[20286],
nusc.ego_pose[20295],
nusc.ego_pose[20288],
nusc.ego_pose[20379],
nusc.ego_pose[20326],
nusc.ego_pose[20332],
nusc.ego_pose[20328],
nusc.ego_pose[20419],
nusc.ego_pose[20369],
nusc.ego_pose[20374],


]


accumulated_pcd = o3d.geometry.PointCloud()
threshold_distance = 2.0  # 1メートルの閾値

for file_path, pose in zip(file_paths, poses):
    # Load point cloud data
    pcd_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)[:, :3]

    # XY平面での距離を計算して1メートル以内の点を除去
    distances_xy = np.linalg.norm(pcd_data[:, :2], axis=1)  # Z軸を無視
    mask = distances_xy > threshold_distance
    pcd_data_filtered = pcd_data[mask]

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data_filtered)

    # Apply calibration transformation
    pcd.transform(calibration_transform)

    # Apply ego pose transformation
    ego_transform = transform_to_matrix(pose['translation'], pose['rotation'])
    pcd.transform(ego_transform)

    # Merge with accumulated point cloud
    accumulated_pcd += pcd

# Visualize the merged point cloud
o3d.visualization.draw_geometries([accumulated_pcd])

# 点群マップの保存
o3d.io.write_point_cloud("map4.pcd", accumulated_pcd)