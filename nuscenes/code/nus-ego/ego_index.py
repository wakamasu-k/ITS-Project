##########################################################################################################
#特定のインデックスのego_pose情報を取得

from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='C://Users/divin/OneDrive - 梅村学園　中京大学//MD/卒業研究/data/nuScenes mini us', verbose=True
)
s1 = nusc.ego_pose[4409]
print(s1)