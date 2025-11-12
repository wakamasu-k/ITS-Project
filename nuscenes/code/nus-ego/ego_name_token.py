#ファイルの名前からego pose tokenを抜き出す　

import json
import os
import pandas as pd

# 解凍されたファイルが保存されているフォルダのパス
extraction_path = 'C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\'  # このパスを実際のフォルダの場所に合わせて変更してください
sample_data_path = 'C:\\Users\\divin\\OneDrive - 梅村学園　中京大学\\MD\\卒業研究\\nuScenes\\nuScenes mini us\\v1.0-mini\\sample_data.json'  # sample_data.jsonのパスも適宜変更してください

# フォルダ内のファイル名をリストアップします
extracted_file_names = os.listdir(extraction_path)

# sample_data.jsonを読み込みます
with open(sample_data_path, 'r') as file:
    sample_data = json.load(file)

# ファイル名に対応するego_pose_tokenを検索します
file_name_to_ego_pose_token = {}
for file_name in extracted_file_names:
    for record in sample_data:
        # 大文字小文字を区別せずにファイル名が一致するか確認します
        if record['filename'].lower().endswith(file_name.lower()):
            file_name_to_ego_pose_token[file_name] = record['ego_pose_token']
            break

# Pandasの表示設定を変更する
pd.set_option('display.max_columns', None)  # 列の最大表示数
pd.set_option('display.max_rows', None)     # 行の最大表示数
pd.set_option('display.width', None)        # 行の幅
pd.set_option('display.max_colwidth', None) # 列の最大幅

# すべてのego_pose_tokenを表示
for file_name, ego_pose_token in file_name_to_ego_pose_token.items():
    print(f"{ego_pose_token}")


# 結果をDataFrameに変換し、表示します
df_tokens = pd.DataFrame(list(file_name_to_ego_pose_token.items()), columns=['File Name', 'Ego Pose Token'])
print(df_tokens)



# 結果をDataFrameに変換し、表示します
#df_tokens = pd.DataFrame(list(file_name_to_ego_pose_token.items()), columns=['File Name', 'Ego Pose Token'])
#print(df_tokens)