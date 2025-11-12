import numpy as np
import copy
import open3d as o3d
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

# =============================================================================
# 1. バウンディングボックスを拡大するユーティリティ
# =============================================================================
def inflate_box(original_box: Box, scale_factor: float = 1.2) -> Box:
    box_copy = copy.deepcopy(original_box)
    box_copy.wlh = box_copy.wlh * scale_factor
    return box_copy

# =============================================================================
# 2. バウンディングボックス描画用
# =============================================================================
def create_open3d_bbox_lineset(box: Box) -> o3d.geometry.LineSet:
    corners_3d = box.corners().T
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_3d)
    line_set.lines  = o3d.utility.Vector2iVector(edges)
    colors = [[1.0, 0.0, 0.0] for _ in edges]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# =============================================================================
# 3. Box内部の点を判定
# =============================================================================
def points_in_box_3d(box: Box, points_xyz: np.ndarray) -> np.ndarray:
    center = box.center
    orientation = box.orientation
    w, l, h = box.wlh
    points_shifted = points_xyz - center[None, :]
    R_inv = orientation.inverse.rotation_matrix
    points_in_box_coords = points_shifted @ R_inv.T

    half_w, half_l, half_h = w/2.0, l/2.0, h/2.0
    mask_x = (np.abs(points_in_box_coords[:, 0]) <= half_w)
    mask_y = (np.abs(points_in_box_coords[:, 1]) <= half_l)
    mask_z = (np.abs(points_in_box_coords[:, 2]) <= half_h)
    return mask_x & mask_y & mask_z

# =============================================================================
# 4. LiDARデータ読み込み
# =============================================================================
def load_lidar_data(filepath: str) -> np.ndarray:
    pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    return pc[:, :4]  # (x, y, z, intensity)

# =============================================================================
# 5. 座標変換行列を用いた点群変換
# =============================================================================
def transform_points(points: np.ndarray, mat_4x4: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    xyz1 = np.concatenate([points[:, :3], ones], axis=1)
    xyz1_tf = (mat_4x4 @ xyz1.T).T
    return np.column_stack((xyz1_tf[:, :3], points[:, 3]))

# =============================================================================
# 6. カテゴリ判定（vehicle/human/movable_object）
# =============================================================================
def is_dynamic_category(category_name: str) -> bool:
    top_level = category_name.split('.')[0]
    return top_level in {'vehicle', 'human', 'movable_object'}

# =============================================================================
# 7. 「本当に動いている」かどうか (属性のmovingをチェック)
# =============================================================================
def is_really_moving(box_token: str, nusc: NuScenes) -> bool:
    """
    Boxに対応するannotationを取得し、attributeに 'moving' が含まれていればTrue
    (例: 'vehicle.moving', 'pedestrian.moving')
    """
    ann_rec = nusc.get('sample_annotation', box_token)
    attr_list = ann_rec['attribute_tokens']
    if len(attr_list) == 0:
        return False
    for atok in attr_list:
        aname = nusc.get('attribute', atok)['name']
        if 'moving' in aname:
            return True
    return False

# =============================================================================
# 8. 指定したサンプルの6台カメラ画像を一括表示 (2x3モザイク)
# =============================================================================
def show_all_camera_images(nusc: NuScenes, sample_idx: int):
    """
    指定したsample_idxのサンプルに含まれる6台カメラ画像を取得し、
    2行×3列でモザイク表示を行う。
    """
    # 6台のカメラチャンネル
    camera_channels = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ]

    my_sample = nusc.sample[sample_idx]
    images = []
    for cam in camera_channels:
        cam_token = my_sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_token)
        img_path = nusc.get_sample_data_path(cam_data['token'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        images.append(img)

    if len(images) == 0:
        print("No camera images found.")
        return

    # 2行×3列に並べる。適宜リサイズ
    resize_w, resize_h = 640, 360
    resized_imgs = [cv2.resize(im, (resize_w, resize_h)) for im in images]

    # もし6枚未満なら黒画像で埋める
    while len(resized_imgs) < 6:
        resized_imgs.append(np.zeros((resize_h, resize_w, 3), dtype=np.uint8))

    # row1 = [0,1,2], row2 = [3,4,5]
    row1 = np.hstack(resized_imgs[:3])
    row2 = np.hstack(resized_imgs[3:6])
    mosaic = np.vstack([row1, row2])

    cv2.imshow("All Camera Images (2x3)", mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# メイン処理
# =============================================================================
if __name__ == "__main__":
    # nuScenesの初期化
    nusc = NuScenes(version='v1.0-mini', dataroot='nuscenes', verbose=True)
    
    # 試したいサンプル番号
    sample_idx = 108

    my_sample = nusc.sample[sample_idx]
    lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])

    # ego_pose と calibrated_sensor
    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_pose  = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # LiDAR生データ読み込み
    lidar_file_path = nusc.get_sample_data_path(lidar_data['token'])
    lidar_points = load_lidar_data(lidar_file_path)

    # LiDAR座標 -> 車両座標
    cs_rot   = Quaternion(lidar_calib['rotation']).rotation_matrix
    cs_trans = np.array(lidar_calib['translation'])
    sensor_to_vehicle = np.eye(4)
    sensor_to_vehicle[:3,:3] = cs_rot
    sensor_to_vehicle[:3, 3] = cs_trans

    # 車両座標 -> 世界座標
    ego_rot = Quaternion(lidar_pose['rotation']).rotation_matrix
    ego_trans = np.array(lidar_pose['translation'])
    vehicle_to_world = np.eye(4)
    vehicle_to_world[:3,:3] = ego_rot
    vehicle_to_world[:3, 3] = ego_trans

    # 変換
    lidar_in_vehicle = transform_points(lidar_points, sensor_to_vehicle)
    lidar_in_world   = transform_points(lidar_in_vehicle, vehicle_to_world)

    # 点群をグレースケール化
    intensities = lidar_in_world[:, 3]
    min_i, max_i = intensities.min(), intensities.max()
    denom = (max_i - min_i) if (max_i - min_i) != 0 else 1e-6
    norm_i = (intensities - min_i)/denom
    colors_gray = np.stack([norm_i, norm_i, norm_i], axis=1)

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(lidar_in_world[:, :3])
    pcd_original.colors = o3d.utility.Vector3dVector(colors_gray)

    # バウンディングボックス読み込み
    boxes_world = nusc.get_boxes(lidar_data['token'])
    scale_factor = 2.0

    # 1) オリジナル点群 + 拡大Box(動的でmovingのみ) 可視化
    linesets = []
    for box in boxes_world:
        if is_dynamic_category(box.name) and is_really_moving(box.token, nusc):
            inflated_box = inflate_box(box, scale_factor)
            ls = create_open3d_bbox_lineset(inflated_box)
        else:
            ls = create_open3d_bbox_lineset(box)
        linesets.append(ls)

    o3d.visualization.draw_geometries([pcd_original] + linesets)

    # 2) 点群削除 (動的+movingのみ)
    filtered_points = lidar_in_world.copy()
    for box in boxes_world:
        if is_dynamic_category(box.name) and is_really_moving(box.token, nusc):
            inflated_box = inflate_box(box, scale_factor)
            mask = points_in_box_3d(inflated_box, filtered_points[:, :3])
            filtered_points = filtered_points[~mask]

    intensities_f = filtered_points[:, 3]
    min_f, max_f  = intensities_f.min(), intensities_f.max()
    denom_f       = (max_f - min_f) if (max_f - min_f) != 0 else 1e-6
    norm_i_f      = (intensities_f - min_f) / denom_f
    colors_gray_f = np.stack([norm_i_f, norm_i_f, norm_i_f], axis=1)

    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points[:, :3])
    pcd_filtered.colors = o3d.utility.Vector3dVector(colors_gray_f)

    # 表示 (削除後の点群)
    o3d.visualization.draw_geometries([pcd_filtered])

    # 3) カメラ画像 (6台) の表示
    show_all_camera_images(nusc, sample_idx)


