import numpy as np
import copy
import open3d as o3d
import time
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box

# ----------------------------------------------------------------------------
# ユーティリティ関数
# ----------------------------------------------------------------------------
def load_lidar_data(filepath: str) -> np.ndarray:
    pc = np.fromfile(filepath, dtype=np.float32).reshape(-1,5)  # x,y,z,i,time
    return pc[:,:4]  # x,y,z,intensity

def transform_points(points: np.ndarray, mat_4x4: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0],1), dtype=np.float32)
    xyz1 = np.hstack([points[:,:3], ones])  # (N,4)
    xyz1_tf = (mat_4x4 @ xyz1.T).T
    return np.hstack([xyz1_tf[:,:3], points[:,3:4]])

def is_dynamic_category(cat_name:str)->bool:
    """
    vehicle/human/movable_object は全部削除対象 (属性は無視)。
    """
    top = cat_name.split('.')[0]
    return top in {'vehicle','human','movable_object'}

def points_in_box_3d(box: Box, points_xyz: np.ndarray)->np.ndarray:
    c = box.center
    o = box.orientation
    w,l,h = box.wlh
    shifted = points_xyz - c[None,:]
    R_inv = o.inverse.rotation_matrix
    coords = shifted @ R_inv.T
    half_w, half_l, half_h = w/2.0, l/2.0, h/2.0
    mask_x = np.abs(coords[:,0])<=half_w
    mask_y = np.abs(coords[:,1])<=half_l
    mask_z = np.abs(coords[:,2])<=half_h
    return mask_x & mask_y & mask_z

def inflate_box(box: Box, scale_factor: float=1.0)->Box:
    box_copy = copy.deepcopy(box)
    box_copy.wlh = box_copy.wlh * scale_factor
    return box_copy

def create_open3d_bbox_lineset(box: Box, color=(1.0,0.0,0.0)) -> o3d.geometry.LineSet:
    corners_3d = box.corners().T
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_3d)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    col_arr = [color]*len(edges)
    ls.colors= o3d.utility.Vector3dVector(col_arr)
    return ls

# -----------------------------
# 色付け関数
# -----------------------------
def apply_reflectance_to_colors(reflectance: np.ndarray)->np.ndarray:
    """
    反射強度をグレースケール化
    """
    max_reflectance_factor = 0.5
    min_reflectance_factor = 1.2
    max_r = np.max(reflectance)*max_reflectance_factor
    min_r = np.min(reflectance)*min_reflectance_factor
    norm_r= (reflectance - min_r)/(max_r - min_r)
    norm_r= np.clip(norm_r,0,1)
    return np.stack([norm_r,norm_r,norm_r], axis=1)

def apply_zcolor(points_xyz: np.ndarray, z_min=None, z_max=None)->np.ndarray:
    """
    Z値(高さ)に応じてグラデーション: R=高い, B=低い
    R=norm_z, G=0, B=1-norm_z
    """
    z_vals = points_xyz[:,2]
    if z_min is None:
        z_min = np.min(z_vals)
    if z_max is None:
        z_max = np.max(z_vals)
    denom= (z_max - z_min) if (z_max>z_min) else 1e-6
    norm_z= (z_vals - z_min)/denom
    norm_z= np.clip(norm_z, 0,1)

    R= norm_z
    G= np.zeros_like(norm_z)
    B= 1.0 - norm_z
    return np.stack([R,G,B], axis=1)

def apply_xcolor(points_xyz: np.ndarray, x_min=None, x_max=None)->np.ndarray:
    """
    X値に応じてグラデーション: R=大きい, B=小さい
    """
    x_vals = points_xyz[:,0]
    if x_min is None:
        x_min = np.min(x_vals)
    if x_max is None:
        x_max = np.max(x_vals)
    denom= (x_max - x_min) if (x_max> x_min) else 1e-6
    norm_x= (x_vals - x_min)/denom
    norm_x= np.clip(norm_x,0,1)

    R= norm_x
    G= np.zeros_like(norm_x)
    B= 1.0 - norm_x
    return np.stack([R,G,B], axis=1)

def apply_ycolor(points_xyz: np.ndarray, y_min=None, y_max=None)->np.ndarray:
    """
    Y値に応じてグラデーション: R=大きい, B=小さい
    """
    y_vals = points_xyz[:,1]
    if y_min is None:
        y_min = np.min(y_vals)
    if y_max is None:
        y_max = np.max(y_vals)
    denom= (y_max - y_min) if (y_max> y_min) else 1e-6
    norm_y= (y_vals - y_min)/denom
    norm_y= np.clip(norm_y,0,1)

    R= norm_y
    G= np.zeros_like(norm_y)
    B= 1.0 - norm_y
    return np.stack([R,G,B], axis=1)

# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
if __name__=="__main__":

    # ユーザー設定
    nusc = NuScenes(version='v1.0-mini',
                    dataroot='C:/Users/divin/Downloads/nuScenes',
                    verbose=True)

    start_frame = 39
    end_frame   = 78
    threshold_distance = 1.0
    bounding_box_scale = 3.0

    # -------------------------------
    # 色付けモード
    #   "reflectance" / "height" / "xaxis" / "yaxis"
    # -------------------------------
    color_mode = "reflectance"

    # (A) Visualizerを初期化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Accumulate Map (Remove Past BBox)")

    # 累積用のPointCloud (過去フレーム含む)
    accumulated_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(accumulated_pcd)

    # 前フレームのバウンディングボックスを管理 (あとで消すため)
    prev_bbox_list = []

    # 1フレームあたりの待機 (FPS的な)
    fps = 2.0
    frame_delay = 1.0/fps

    for sample_idx in range(start_frame, end_frame+1):
        print(f"--- Processing frame={sample_idx} ---")

        # LiDAR読み込み & 近すぎる点除外
        my_sample = nusc.sample[sample_idx]
        lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])

        calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        ego   = nusc.get('ego_pose',          lidar_data['ego_pose_token'])

        s2v = np.eye(4)
        s2v[:3,:3] = Quaternion(calib['rotation']).rotation_matrix
        s2v[:3, 3] = np.array(calib['translation'])

        v2w = np.eye(4)
        v2w[:3,:3] = Quaternion(ego['rotation']).rotation_matrix
        v2w[:3, 3] = np.array(ego['translation'])

        lidar_filepath = nusc.get_sample_data_path(lidar_data['token'])
        lidar_points = load_lidar_data(lidar_filepath)

        dist_xy = np.linalg.norm(lidar_points[:,:2], axis=1)
        mask = (dist_xy>threshold_distance)
        lidar_points_filtered = lidar_points[mask]

        # 座標変換
        p_world = transform_points(lidar_points_filtered, s2v)
        p_world = transform_points(p_world, v2w)

        # 動的カテゴリBoxで消去
        boxes_world = nusc.get_boxes(lidar_data['token'])
        for box in boxes_world:
            if is_dynamic_category(box.name):
                scaled_box = inflate_box(box, bounding_box_scale)
                mask_in = points_in_box_3d(scaled_box, p_world[:,:3])
                p_world = p_world[~mask_in]

        # 累積点群に追加
        old_points_np = np.asarray(accumulated_pcd.points)
        old_colors_np = np.asarray(accumulated_pcd.colors)
        new_points_np = p_world[:,:3]

        # ---- 色付けモードに応じた処理 ----
        if color_mode=="reflectance":
            # 反射強度でグレースケール
            new_colors_np = apply_reflectance_to_colors(p_world[:,3])
        elif color_mode=="height":
            # z軸カラー
            new_colors_np = apply_zcolor(new_points_np)
        elif color_mode=="xaxis":
            # x軸カラー
            new_colors_np = apply_xcolor(new_points_np)
        elif color_mode=="yaxis":
            # y軸カラー
            new_colors_np = apply_ycolor(new_points_np)
        else:
            # デフォルト: reflectance
            new_colors_np = apply_reflectance_to_colors(p_world[:,3])

        if old_points_np.size>0:
            all_points = np.vstack([old_points_np, new_points_np])
            all_colors = np.vstack([old_colors_np, new_colors_np])
        else:
            all_points = new_points_np
            all_colors = new_colors_np

        accumulated_pcd.points = o3d.utility.Vector3dVector(all_points)
        accumulated_pcd.colors = o3d.utility.Vector3dVector(all_colors)

        # (B) 前フレームのBBoxを消す
        for bbox_geom in prev_bbox_list:
            vis.remove_geometry(bbox_geom)
        prev_bbox_list.clear()

        # (C) 今フレームのBBoxを追加
        this_frame_bbox_list = []
        for box in boxes_world:
            scaled_box = inflate_box(box, bounding_box_scale)
            ls = create_open3d_bbox_lineset(scaled_box, color=(1,0,0))
            this_frame_bbox_list.append(ls)
            vis.add_geometry(ls)

        prev_bbox_list = this_frame_bbox_list

        # Open3Dに更新を通知
        vis.update_geometry(accumulated_pcd)
        for bbox_geom in this_frame_bbox_list:
            vis.update_geometry(bbox_geom)

        vis.poll_events()
        vis.update_renderer()

        # 次フレームまで待機
        time.sleep(frame_delay)

    print("Finished. Close the window to end.")
    # ウィンドウを閉じるまで待ちたい場合は user input or loop
    vis.destroy_window()
