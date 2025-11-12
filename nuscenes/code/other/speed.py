import numpy as np
from nuscenes.nuscenes import NuScenes

# -------------------------------------------------------------------
# ユーザが指定した「サンプル番号」
sample_idx_1 = 161
sample_idx_2 = 201
dataroot_path = "C:\\Users\\shimizu.k\\Downloads\\nuScenes"  # 例
# -------------------------------------------------------------------

# nuScenesオブジェクトを初期化
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot_path, verbose=True)

# sampleをインデックス指定で取得
sample1 = nusc.sample[sample_idx_1]
sample2 = nusc.sample[sample_idx_2]

# ここではLIDAR_TOPを使って例示（CAM_FRONTやRADAR_FRONTなども可）
lidar_token_1 = sample1['data']['LIDAR_TOP']
lidar_token_2 = sample2['data']['LIDAR_TOP']

# sample_dataレコードを取得
sd1 = nusc.get('sample_data', lidar_token_1)
sd2 = nusc.get('sample_data', lidar_token_2)

# ego_poseを取得
ego_pose_1 = nusc.get('ego_pose', sd1['ego_pose_token'])
ego_pose_2 = nusc.get('ego_pose', sd2['ego_pose_token'])

# 位置(x, y, z)とタイムスタンプを取り出す
pos_1 = np.array(ego_pose_1['translation'])
t_1 = ego_pose_1['timestamp']  # ナノ秒

pos_2 = np.array(ego_pose_2['translation'])
t_2 = ego_pose_2['timestamp']

# 時間差(秒)
dt = (t_2 - t_1) / 1e6  # ナノ秒 -> 秒

# 位置差(m)
dpos = pos_2 - pos_1

# 速度ベクトル[m/s] と その大きさ
if dt > 0:
    velocity_vector = dpos / dt
    speed_m_s = np.linalg.norm(velocity_vector)
else:
    velocity_vector = np.array([0.0, 0.0, 0.0])
    speed_m_s = 0.0

# m/s を km/h に変換
speed_kmh = speed_m_s * 3.6

# 結果を表示
print(f"Sample[{sample_idx_1}] -> Sample[{sample_idx_2}]")
print(f"  position_1: {pos_1}, time_1: {t_1}")
print(f"  position_2: {pos_2}, time_2: {t_2}")
print(f"  velocity_vector (m/s): {velocity_vector}")
print(f"  speed (m/s):          {speed_m_s}")
print(f"  speed (km/h):          {speed_kmh}")