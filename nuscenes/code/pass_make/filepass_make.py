import os

# ベースのフォルダパス
base_folder_path = "C:\\Users\\divin\\Downloads\\scene4_sgp\\lidar\\lidar"

# 指定フォルダ内のすべてのファイル名を取得
file_names = os.listdir(base_folder_path)

# 完全なファイルパスのリストを生成
full_file_paths = [os.path.join(base_folder_path, file_name) for file_name in file_names]

# 完全なファイルパスを出力
for file_path in full_file_paths:
    print(file_path)
