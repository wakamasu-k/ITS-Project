import numpy as np
from nuscenes.nuscenes import NuScenes

start_idx = 0
end_idx   = 38
dataroot_path = "C:/Users/shimizu.k/Downloads/nuScenes"

nusc = NuScenes(version='v1.0-mini', dataroot=dataroot_path, verbose=True)

speeds = []

for i in range(start_idx, end_idx):
    if i+1 >= len(nusc.sample):
        break
    
    s_i   = nusc.sample[i]
    s_i1  = nusc.sample[i+1]
    
    # sample同士のtimestamp
    t_i   = s_i['timestamp']   # マイクロ秒
    t_i1  = s_i1['timestamp']  # マイクロ秒
    dt = (t_i1 - t_i)/1e6  # 秒
    
    # 位置は s_i の LIDAR_TOP からエゴ車両位置を取得
    #   => 例: ego_pose
    lidar_token_i = s_i['data']['LIDAR_TOP']
    sd_i = nusc.get('sample_data', lidar_token_i)
    ego_pose_i = nusc.get('ego_pose', sd_i['ego_pose_token'])
    pos_i = np.array(ego_pose_i['translation'])
    
    lidar_token_i1= s_i1['data']['LIDAR_TOP']
    sd_i1= nusc.get('sample_data', lidar_token_i1)
    ego_pose_i1= nusc.get('ego_pose', sd_i1['ego_pose_token'])
    pos_i1= np.array(ego_pose_i1['translation'])
    
    dpos= pos_i1 - pos_i
    dist= np.linalg.norm(dpos)
    
    speed_m_s= 0.0
    if dt>0:
        speed_m_s= dist/dt
    speed_km_h= speed_m_s*3.6
    
    speeds.append(speed_m_s)
    
    print(f"Frames {i}->{i+1}, dt={dt:.3f}s, dist={dist:.3f}m => {speed_m_s:.3f}m/s ({speed_km_h:.3f}km/h)")

if len(speeds)>0:
    avg_m_s= np.mean(speeds)
    avg_km_h= avg_m_s*3.6
    print("---------------------")
    print(f"Average speed over {len(speeds)} intervals: {avg_m_s:.3f} m/s, {avg_km_h:.3f} km/h")
