# 依存: nuscenes-devkit, pyquaternion, numpy
import os, numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
import cv2

# --- パラメータ ---
DATA_ROOT = r"nuscenes"
VERSION = "v1.0-mini"
# 使用する LiDAR センサ名（データ次第）
SENSOR_NAME = "LIDAR_TOP"   # 例: "LIDAR_TOP"（デフォルト）. "LIDAR_FRONT" 等があれば指定可
# 使用するカメラ名（投影例）
CAM_NAME = "CAM_FRONT"  # CAM_FRONT 等

# --- LiDAR点群をsensor->ego->global に変換する関数（sensor名を決めて呼ぶ） ---
def get_lidar_in_ego_or_global(nusc, lidar_token, to_global=True):
    sd = nusc.get('sample_data', lidar_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])

    lidar_path = os.path.join(nusc.dataroot, sd['filename'])
    pc = LidarPointCloud.from_file(lidar_path)  # shape: (4, N)

    # sensor -> ego
    mat_sensor2ego = transform_matrix(cs['translation'],
                                      Quaternion(cs['rotation']),
                                      inverse=False)
    pc.transform(mat_sensor2ego)

    if to_global:
        # ego -> global
        mat_ego2global = transform_matrix(ep['translation'],
                                          Quaternion(ep['rotation']),
                                          inverse=False)
        pc.transform(mat_ego2global)

    return pc.points.T  # (N,4)

# --- カメラ座標系の3D点をEGOに変換する関数（例えば外れた3D点をカメラから得た場合） ---
def camera_to_ego_points(nusc, camera_sample_data_token, points_cam):
    """
    points_cam: (N,3) array in camera coordinate (x,y,z) (camera frame)
    returns: points in ego frame (N,3)
    """
    sd = nusc.get('sample_data', camera_sample_data_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])

    # camera -> ego: use calibrated_sensor rotation/translation (cs)
    mat_cam2ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
    # ego -> global if needed: transform_matrix(ep['translation'], Quaternion(ep['rotation']), inverse=False)

    # convert points to homogeneous
    ones = np.ones((points_cam.shape[0],1))
    homo = np.concatenate([points_cam, ones], axis=1).T  # (4,N)
    transformed = mat_cam2ego.dot(homo)  # (4,N)
    return transformed[:3,:].T  # (N,3) in ego

# --- LiDAR点をカメラ画像に投影する（深度付き投影） ---
def project_lidar_to_image(nusc, lidar_points_ego, camera_sample_data_token):
    """
    lidar_points_ego: (N,3) points in EGO frame
    returns:
       uv: (2, M) image pixel coords
       depth: (M,) distances
       indices: indices of lidar_points that were projected (M subset of N)
    """
    sd = nusc.get('sample_data', camera_sample_data_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    # camera intrinsic: fx,fy,cx,cy stored in calibrated_sensor['camera_intrinsic']
    K = np.array(cs['camera_intrinsic']).reshape(3,3)

    # ego -> camera: need inverse of camera->ego (i.e., ego->camera)
    mat_cam2ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
    mat_ego2cam = np.linalg.inv(mat_cam2ego)

    # convert lidar points (ego) to camera frame
    pts = lidar_points_ego.T  # (3,N)
    ones = np.ones((1, pts.shape[1]))
    homo = np.vstack([pts, ones])  # (4,N)
    cam_pts = mat_ego2cam.dot(homo)  # (4,N)
    cam_pts = cam_pts[:3,:]  # (3,N)

    # keep points in front of camera (z>0)
    z = cam_pts[2,:]
    mask = z > 0.01
    cam_pts = cam_pts[:,mask]

    # project using intrinsics
    uv = K.dot(cam_pts)           # (3,M)
    uv[:2,:] /= uv[2:3,:]         # normalize
    uvs = uv[:2,:]                # (2,M)
    depths = cam_pts[2,:]
    indices = np.where(mask)[0]
    return uvs, depths, indices

# --- 例: 使い方 ---
nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)
scene = nusc.scene[0]
sample_token = scene['first_sample_token']
sample = nusc.get('sample', sample_token)

# LiDARトークン探し（センサ名で選べるように）
lidar_token = sample['data'].get(SENSOR_NAME)
if lidar_token is None:
    print(f"センサ {SENSOR_NAME} がこのサンプルに存在しません。available keys: {list(sample['data'].keys())}")
else:
    pts = get_lidar_in_ego_or_global(nusc, lidar_token, to_global=False)  # ego frame
    # ptsは (N,4) -> [:,:3]が座標

# カメラ投影の例
cam_token = sample['data'].get(CAM_NAME)
if cam_token is not None:
    uvs, depths, inds = project_lidar_to_image(nusc, pts[:,:3], cam_token)
    # 画像への描画例
    img_path = os.path.join(nusc.dataroot, nusc.get('sample_data', cam_token)['filename'])
    img = cv2.imread(img_path)
    for i in range(uvs.shape[1]):
        u,v = int(uvs[0,i]), int(uvs[1,i])
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u,v), 1, (0,255,0), -1)
    cv2.imwrite("projected.png", img)
nusc = NuScenes(version='v1.0-mini', dataroot=r'nuscenes', verbose=True)
print("Loaded scenes:", len(nusc.scene))
