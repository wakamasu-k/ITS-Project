#ファイルの名前からcalibrated_sensor_tokenを抜き出す　

import json
import os
import pandas as pd

# 解凍されたファイルが保存されているフォルダのパス
extraction_path = 'C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\first'  # このパスを実際のフォルダの場所に合わせて変更してください
sample_data_path = 'C:\\Users\\divin\\OneDrive - 梅村学園　中京大学\\MD\\卒業研究\\data\\nuScenes mini us\\v1.0-mini\\sample_data.json'  # sample_data.jsonのパスも適宜変更してください

# フォルダ内のファイル名をリストアップします
extracted_file_names = os.listdir(extraction_path)

# sample_data.jsonを読み込みます
with open(sample_data_path, 'r') as file:
    sample_data = json.load(file)

# ファイル名に対応するcalibrated_sensor_tokenを検索します
file_name_to_calibrated_sensor_token = {}
for file_name in extracted_file_names:
    for record in sample_data:
        # 大文字小文字を区別せずにファイル名が一致するか確認します
        if record['filename'].lower().endswith(file_name.lower()):
            file_name_to_calibrated_sensor_token[file_name] = record['calibrated_sensor_token']
            break

# Pandasの表示設定を変更する
pd.set_option('display.max_columns', None)  # 列の最大表示数
pd.set_option('display.max_rows', None)     # 行の最大表示数
pd.set_option('display.width', None)        # 行の幅
pd.set_option('display.max_colwidth', None) # 列の最大幅

# すべてのcalibrated_sensor_tokenを表示
for file_name, calibrated_sensor_token in file_name_to_calibrated_sensor_token.items():
    print(f"{calibrated_sensor_token}")


# 結果をDataFrameに変換し、表示します
df_tokens = pd.DataFrame(list(file_name_to_calibrated_sensor_token.items()), columns=['File Name', 'calibrated_sensor_token'])
print(df_tokens)
























"""from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='C:\\Users\\divin\\OneDrive - 梅村学園　中京大学\\MD\\卒業研究\\data\\nuScenes mini us', verbose=True)

my_scene = nusc.scene[1]
#print(my_scene)S
ego_record = nusc.ego_pose[10540]
#calibrated = nusc.calibrated_sensor[17]
print(ego_record)
#print(calibrated)

#token = "603a96c23db34acf8b5953d958df293d"  # 調べたいトークンを指定

#for i, calibrated in enumerate(nusc.calibrated_sensor):
   # if calibrated['token'] == token:
       # print(f"The index i for the token '{token}' is: {i}")
        #break
    
#for i, ego_record in enumerate(nusc.ego_pose):
   # if ego_record['token'] == token:
       # print(f"The index i for the token '{token}' is: {i}")
        #break
        
        
        
        
"""

