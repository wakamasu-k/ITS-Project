#対数スケーリング
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from matplotlib import cm
from nuscenes.nuscenes import NuScenes

# Datasetのロード
nusc = NuScenes(version='v1.0-mini', dataroot='C:\\Users\\divin\\OneDrive - 梅村学園　中京大学\\MD\\卒業研究\\data\\nuScenes mini us', verbose=True)

# Helper function to create a transformation matrix from translation and rotation
def transform_to_matrix(translation, rotation):
    q = Quaternion(rotation)
    matrix = np.eye(4)
    matrix[:3, :3] = q.rotation_matrix
    matrix[:3, 3] = translation
    return matrix

# Get calibration data
calibration_data = nusc.calibrated_sensor[41]
calibration_transform = transform_to_matrix(calibration_data['translation'], calibration_data['rotation'])


# File paths and corresponding poses
file_paths = [
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385092150099.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385092700299.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385093200183.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385093650312.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385094150195.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385094550004.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385094949783.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385095449675.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385095949554.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385096399680.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385096900138.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385097400558.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385097901019.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385098400887.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385098900804.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385099400097.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385099899436.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385100398781.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385100898103.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385101398009.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385101947644.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385102447561.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385102947981.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385103447886.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385103948321.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385104448758.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385104949764.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385105450180.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385105950634.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385106451625.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385106952072.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385107452513.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385107953148.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385108451734.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385108951051.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385109400618.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385109899957.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385110449087.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385110948962.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385111448839.pcd.bin",
    "C:\\Users\\divin\\Downloads\\scene2\\LiDAR\\pcd.bin\\n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385111949270.pcd.bin",
]

poses = [
    nusc.ego_pose[10997],
    nusc.ego_pose[10923],
    nusc.ego_pose[10899],
    nusc.ego_pose[10912],
    nusc.ego_pose[10951],
    nusc.ego_pose[10904],
    nusc.ego_pose[10896],
    nusc.ego_pose[10816],
    nusc.ego_pose[10805],
    nusc.ego_pose[10826],
    nusc.ego_pose[10875],
    nusc.ego_pose[10742],
    nusc.ego_pose[10746],
    nusc.ego_pose[10747],
    nusc.ego_pose[10750],
    nusc.ego_pose[10759],
    nusc.ego_pose[10757],
    nusc.ego_pose[10767],
    nusc.ego_pose[10718],
    nusc.ego_pose[10725],
    nusc.ego_pose[10702],
    nusc.ego_pose[10663],
    nusc.ego_pose[10660],
    nusc.ego_pose[10707],
    nusc.ego_pose[10689],
    nusc.ego_pose[10594],
    nusc.ego_pose[10632],
    nusc.ego_pose[10573],
    nusc.ego_pose[10590],
    nusc.ego_pose[10589],
    nusc.ego_pose[10591],
    nusc.ego_pose[10626],
    nusc.ego_pose[10634],
    nusc.ego_pose[10572],
    nusc.ego_pose[10576],
    nusc.ego_pose[10540],
    nusc.ego_pose[10435],
    nusc.ego_pose[10498],
    nusc.ego_pose[10503],
    nusc.ego_pose[11010],
    nusc.ego_pose[11062],
]


accumulated_pcd = o3d.geometry.PointCloud()
threshold_distance = 2.0  # 2メートルの閾値

for file_path, pose in zip(file_paths, poses):
    # Load point cloud data
    pcd_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    points = pcd_data[:, :3]  # XYZ座標
    intensities = pcd_data[:, 3]  # 反射強度

    # XY平面での距離を計算して閾値以内の点を除去
    distances_xy = np.linalg.norm(points[:, :2], axis=1)  # Z軸を無視
    mask = distances_xy > threshold_distance
    points_filtered = points[mask]
    intensities_filtered = intensities[mask]

    # 反射強度を色にマッピング
    intensities_log = np.log(intensities_filtered + 1)  # 対数スケーリング
    normalized_intensities = (intensities_log - intensities_log.min()) / (intensities_log.max() - intensities_log.min())
    colors_filtered = cm.jet(normalized_intensities)[:, :3]

    # 点群データの作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered)
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
    
    # キャリブレーション行列の適用
    pcd.transform(calibration_transform)
    
    # ego_poseの適用
    translation_pose = pose['translation']
    rotation_matrix_pose = Quaternion(pose['rotation']).rotation_matrix
    transformation_pose = np.eye(4)
    transformation_pose[:3, :3] = rotation_matrix_pose
    transformation_pose[:3, 3] = translation_pose
    
    pcd.transform(transformation_pose)

    # 点群マップに追加
    accumulated_pcd += pcd

# 点群マップの保存
o3d.io.write_point_cloud("map2_log_scail.pcd", accumulated_pcd)