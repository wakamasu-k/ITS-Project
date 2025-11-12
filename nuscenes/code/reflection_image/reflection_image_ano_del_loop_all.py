import numpy as np
import copy
import open3d as o3d
import cv2
import os
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

# =============================================================================
# 1. バウンディングボックス関連
# =============================================================================
def points_in_box_3d(box: Box, points_xyz: np.ndarray) -> np.ndarray:
    """
    指定Box内に含まれる点を判定するマスク(True/False)を返す。
    """
    center = box.center
    orientation = box.orientation
    w, l, h = box.wlh

    # 点群をBoxの中心へ平行移動
    points_shifted = points_xyz - center[None, :]

    # 逆回転行列で点を回転
    R_inv = orientation.inverse.rotation_matrix
    points_in_box_coords = points_shifted @ R_inv.T

    # 軸平行AABBで判定
    half_w, half_l, half_h = w/2.0, l/2.0, h/2.0
    mask_x = np.abs(points_in_box_coords[:,0]) <= half_w
    mask_y = np.abs(points_in_box_coords[:,1]) <= half_l
    mask_z = np.abs(points_in_box_coords[:,2]) <= half_h

    return mask_x & mask_y & mask_z

def is_dynamic_category(category_name: str) -> bool:
    """
    トップレベルが vehicle/human/movable_object なら動的とみなし削除対象。
    （parked, stopped, movingなど属性は無視する）
    """
    top_level = category_name.split('.')[0]
    return top_level in {'vehicle', 'human', 'movable_object'}

def inflate_box(original_box: Box, scale_factor: float) -> Box:
    """
    Boxをdeepcopyして w,l,h を scale_factor倍に拡大(または縮小)する。
    1.0でオリジナル、2.0で2倍大きいBoxなど。
    """
    box_copy = copy.deepcopy(original_box)
    box_copy.wlh = box_copy.wlh * scale_factor
    return box_copy

# =============================================================================
# 2. LiDAR関連 (読み込み・座標変換・反射強度着色)
# =============================================================================
def load_lidar_data(filepath: str) -> np.ndarray:
    """
    binファイルから (x,y,z,intensity) をnumpyで返す
    """
    pc = np.fromfile(filepath, dtype=np.float32).reshape(-1,5)  # x,y,z,i,time
    return pc[:, :4]

def transform_points(points: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    """
    点群 [N,4](xyz,intensity) を4x4行列で一括変換。intensityは維持する。
    """
    ones = np.ones((points.shape[0],1), dtype=np.float32)
    xyz1 = np.concatenate([points[:,:3], ones], axis=1)  # N,4
    xyz1_tf = (matrix_4x4 @ xyz1.T).T  # shape=(N,4)
    return np.hstack([xyz1_tf[:,:3], points[:,3:4]])

def apply_reflectance_to_colors(reflectance: np.ndarray) -> np.ndarray:
    """
    反射強度をグレースケール化して [R,G,B]=[i,i,i] (0~1) で返す。
    """
    max_reflectance_factor = 0.5
    min_reflectance_factor = 1.2

    max_r = np.max(reflectance) * max_reflectance_factor
    min_r = np.min(reflectance) * min_reflectance_factor
    norm_ref = (reflectance - min_r) / (max_r - min_r)
    norm_ref = np.clip(norm_ref, 0, 1)

    colors = np.stack([norm_ref, norm_ref, norm_ref], axis=1)
    return colors

# =============================================================================
# 3. メイン処理
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------
    # (A) ユーザー設定
    # -------------------------------------------------------
    nusc = NuScenes(version='v1.0-mini', dataroot='D:\\Users\\wakamatsu.k\\Desktop\\ITS\\nuscenes', verbose=True)
    
    start_sample_number = 364
    end_sample_number   = 403
    threshold_distance  = 5.0
    bounding_box_scale  = 3.0  # 1.0: オリジナルサイズ, 2.0: 2倍, 0.5: 半分 など

    # カメラ投影の際の出力先
    base_intensity_folder = Path(r"D:\\Users\\wakamatsu.k\\Desktop\\ITS\\nuscenes\\match\\364_403\\intensity_ano_del_all")
    base_intensity_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # (B) 複数フレームのLiDARを世界座標系へ変換→動的カテゴリ削除→結合
    # -------------------------------------------------------
    transformed_points_list = []

    for sample_number in range(start_sample_number, end_sample_number+1):
        my_sample = nusc.sample[sample_number]
        lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])

        # キャリブ情報・ego_pose取得
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_pose  = nusc.get('ego_pose',          lidar_data['ego_pose_token'])

        # 4x4行列
        sensor_to_vehicle = np.eye(4)
        sensor_to_vehicle[:3,:3] = Quaternion(lidar_calib['rotation']).rotation_matrix
        sensor_to_vehicle[:3, 3] = np.array(lidar_calib['translation'])

        vehicle_to_world = np.eye(4)
        vehicle_to_world[:3,:3] = Quaternion(lidar_pose['rotation']).rotation_matrix
        vehicle_to_world[:3, 3] = np.array(lidar_pose['translation'])

        # LiDAR読込
        lidar_filepath = nusc.get_sample_data_path(lidar_data['token'])
        lidar_points   = load_lidar_data(lidar_filepath)

        # 2m以内除外
        dist_xy = np.linalg.norm(lidar_points[:,:2], axis=1)
        mask    = (dist_xy > threshold_distance)
        lidar_points_filtered = lidar_points[mask]

        # 座標変換
        points_world = transform_points(lidar_points_filtered, sensor_to_vehicle)
        points_world = transform_points(points_world, vehicle_to_world)

        # 動的カテゴリのBoxを全部削除 (属性は無視)
        boxes = nusc.get_boxes(lidar_data['token'])
        for box in boxes:
            if is_dynamic_category(box.name):
                # bounding_box_scale倍に拡大/縮小
                scaled_box = inflate_box(box, bounding_box_scale)
                mask_in_box = points_in_box_3d(scaled_box, points_world[:,:3])
                points_world = points_world[~mask_in_box]

        transformed_points_list.append(points_world)

    # 結合
    accumulated_points = np.vstack(transformed_points_list)

    # 全フレーム分を1回だけ保存
    pcd_accum = o3d.geometry.PointCloud()
    pcd_accum.points = o3d.utility.Vector3dVector(accumulated_points[:,:3])
    o3d.io.write_point_cloud("ano_del_all.pcd", pcd_accum)
    print("Saved ano_del_all.pcd (final merged)")

    # -------------------------------------------------------
    # (C) カメラ座標系へ変換 → 反射強度画像の作成
    # -------------------------------------------------------
    # すでに動的カテゴリ削除済みのaccumulated_pointsを用いる
    transformed_accumulated_points_list = []

    for sample_number in range(start_sample_number, end_sample_number+1):
        my_sample = nusc.sample[sample_number]

        camera_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        cam_calib = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        cam_ego   = nusc.get('ego_pose',          camera_data['ego_pose_token'])

        cam_intrinsic = np.array(cam_calib['camera_intrinsic'])
        cam_img_path  = nusc.get_sample_data_path(camera_data['token'])
        cam_img       = cv2.imread(cam_img_path)  # 使わないが一応読み込み
        img_w, img_h  = camera_data['width'], camera_data['height']

        # 世界->cam_ego
        world_cam_ego_rt = np.eye(4)
        world_cam_ego_rt[:3,:3] = Quaternion(cam_ego['rotation']).rotation_matrix.T
        world_cam_ego_tr = np.eye(4)
        world_cam_ego_tr[:3,3] = -np.array(cam_ego['translation'])

        # cam_ego->cam_calib
        world_cam_calib_rt = np.eye(4)
        world_cam_calib_rt[:3,:3] = Quaternion(cam_calib['rotation']).rotation_matrix.T
        world_cam_calib_tr = np.eye(4)
        world_cam_calib_tr[:3,3] = -np.array(cam_calib['translation'])

        # 点群変換
        points_cam = transform_points(accumulated_points, world_cam_ego_tr)
        points_cam = transform_points(points_cam, world_cam_ego_rt)
        points_cam = transform_points(points_cam, world_cam_calib_tr)
        points_cam = transform_points(points_cam, world_cam_calib_rt)

        transformed_accumulated_points_list.append((sample_number, img_w, img_h, points_cam, cam_intrinsic))

    # circle_sizes = [1,2,3,4,5,6]
    circle_sizes = [1,2,3,4,5,6]
    for cs in circle_sizes:
        out_folder = base_intensity_folder / str(cs)
        out_folder.mkdir(parents=True, exist_ok=True)

        for (snum, iw, ih, p_cam, cam_int) in transformed_accumulated_points_list:
            refl = p_cam[:,3]
            colors = apply_reflectance_to_colors(refl)

            # 透視投影行列
            persp_mat = np.zeros((3,4))
            persp_mat[:3,:3] = cam_int

            mask_front = (p_cam[:,2] > 0)
            pts_3d = p_cam[mask_front,:3]
            cols   = colors[mask_front]

            homog_3d = np.hstack([pts_3d, np.ones((pts_3d.shape[0],1))])
            proj_2d  = persp_mat @ homog_3d.T
            proj_2d /= proj_2d[2,:]
            proj_2d  = proj_2d[:2,:].T

            # 奥行きが大きい順に描画
            sort_idx = np.argsort(-pts_3d[:,2])
            proj_2d_sorted = proj_2d[sort_idx]
            cols_sorted    = cols[sort_idx]

            # 真っ黒画像
            reflectance_img = np.zeros((ih, iw, 3), dtype=np.uint8)

            # circle描画
            for (xx,yy), cc in zip(proj_2d_sorted, cols_sorted):
                x_i, y_i = int(xx), int(yy)
                bgr = tuple(int(cc[k]*255) for k in range(3))
                if 0 <= x_i < iw and 0 <= y_i < ih:
                    cv2.circle(reflectance_img, (x_i,y_i), cs, bgr, -1)

            save_path = out_folder / f"Intensity_Image_{snum}.png"
            cv2.imwrite(str(save_path), reflectance_img)
            print(f"[circle_size={cs}] Saved {save_path}")
    
    print("All done.")
    accumulated_pcd = o3d.geometry.PointCloud()
accumulated_pcd.points = o3d.utility.Vector3dVector(accumulated_points[:, :3])

# 点群を可視化
o3d.visualization.draw_geometries([accumulated_pcd])

# 保存ディレクトリを作成して保存
save_dir = Path(r"D:\Users\wakamatsu.k\Desktop\ITS\nuscenes\output_map")
save_dir.mkdir(parents=True, exist_ok=True)
save_pcd_path = save_dir / "map_static_only.pcd"

# 保存
o3d.io.write_point_cloud(str(save_pcd_path), accumulated_pcd)


