#反射強度を読み取ってマップを生成し、反射強度値を正規化し色付け
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
calibration_data = nusc.calibrated_sensor[65]
calibration_transform = transform_to_matrix(calibration_data['translation'], calibration_data['rotation'])

# File paths and corresponding poses
file_paths = [
   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448744447639.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448745047596.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448745547460.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448746047898.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448746548329.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448747047643.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448747547529.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448748047387.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448748547277.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448749047731.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448749547584.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448750047463.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448750547877.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448751048300.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448751548178.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448752048628.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448752549040.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448753048377.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448753547684.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448754047572.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448754547448.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448755047865.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448755548316.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448756050426.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448756547528.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448757048500.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448757548354.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448758048230.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448758547578.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448759047433.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448759546758.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448760047738.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448760548179.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448761048087.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448761548500.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448762048350.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448762547686.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448763047541.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448763547987.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448764047314.pcd.bin",
]

poses = [
   nusc.ego_pose[16833],
   nusc.ego_pose[16844],
   nusc.ego_pose[16838],
   nusc.ego_pose[16866],
   nusc.ego_pose[16789],
   nusc.ego_pose[16884],
   nusc.ego_pose[16877],
   nusc.ego_pose[16906],
   nusc.ego_pose[16828],
   nusc.ego_pose[16928],
   nusc.ego_pose[16915],
   nusc.ego_pose[16946],
   nusc.ego_pose[16865],
   nusc.ego_pose[16968],
   nusc.ego_pose[16956],
   nusc.ego_pose[16989],
   nusc.ego_pose[16904],
   nusc.ego_pose[17009],
   nusc.ego_pose[16995],
   nusc.ego_pose[17030],
   nusc.ego_pose[16941],
   nusc.ego_pose[17049],
   nusc.ego_pose[17034],
   nusc.ego_pose[17070],
   nusc.ego_pose[16978],
   nusc.ego_pose[17091],
   nusc.ego_pose[17071],
   nusc.ego_pose[17109],
   nusc.ego_pose[17016],
   nusc.ego_pose[17128],
   nusc.ego_pose[17134],
   nusc.ego_pose[17146],
   nusc.ego_pose[17054],
   nusc.ego_pose[17169],
   nusc.ego_pose[17174],
   nusc.ego_pose[17105],
   nusc.ego_pose[17125],
   nusc.ego_pose[17240],
   nusc.ego_pose[17280],
   nusc.ego_pose[17141],
]

accumulated_pcd = o3d.geometry.PointCloud()
threshold_distance = 2.0  # 2メートルの閾値

for file_path, pose in zip(file_paths, poses):
    # Load point cloud data
    pcd_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    points = pcd_data[:, :3]  # XYZ座標
    intensities = pcd_data[:, 3]  # 反射強度

    # 反射強度を色情報として使用
    # 反射強度を0〜1の範囲に正規化（必要に応じて調整）
    # 反射強度を色にマッピング
    normalized_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
    colors = cm.jet(normalized_intensities)[:, :3]

    # XY平面での距離を計算して閾値以内の点を除去
    distances_xy = np.linalg.norm(points[:, :2], axis=1)  # Z軸を無視
    mask = distances_xy > threshold_distance
    points_filtered = points[mask]
    colors_filtered = colors[mask]

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered)
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)  # 色を設定

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
o3d.io.write_point_cloud("map3_intensity.pcd", accumulated_pcd)

