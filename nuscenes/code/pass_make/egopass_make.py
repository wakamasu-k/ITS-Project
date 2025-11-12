# ファイルパス
file_path = 'D:\\Users\\wakamatsu.k\\Desktop\\ITS\\nuscenes\\scene4_sgp\\lidar\\ego_index.txt'

# ファイルを開いて各行を読み込む
with open(file_path, 'r') as file:
    indices = file.read().strip().split('\n')

# フォーマットされた文字列を生成して出力する
for index in indices:
    print(f"nusc.ego_pose[{index}],")
