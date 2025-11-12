#####################################################################################
"""
アップロードされたZIPファイルからLiDARデータのファイル名を取得します。
次に、それぞれのLiDARデータのタイムスタンプに最も近いego_pose.jsonのエントリを探し出し、
それぞれのtranslationの値を抽出します。最後に、抽出したtranslationの値を使用して散布図を作成します。

手順は次のとおりです：

ZIPファイルを解凍してLiDARデータのファイル名を取得。
各LiDARデータに対して、最も近いego_pose.jsonのエントリを探し、translationの値を抽出。
抽出したtranslationの値を使用して散布図を作成。

上記の散布図は、各LiDARデータのego_pose.jsonからのtranslation値を示しています。
具体的には、X軸とY軸はそれぞれのLiDARデータのXおよびYのtranslationを示し、色はZのtranslationを示しています。
viridisカラーマップを使用してZのtranslationの値に基づいて色付けを行いました。

"""

import json
import zipfile
import os
import matplotlib.pyplot as plt

# Define the temporary directory for extraction
temp_dir = "C://Users/divin/Download"

# Extract the ZIP file to the temporary directory
with zipfile.ZipFile('C://Users/divin/Downloads/liar_Fh20z.zip', 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Get all LiDAR filenames in the extracted directory
lidar_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
lidar_timestamps = [int(file.split("__")[-1].split(".")[0]) for file in lidar_files]

# Load the ego_pose.json data
with open("C:/Users/divin/OneDrive - 梅村学園　中京大学/MD/卒業研究/data/nuScenes mini us/v1.0-mini/ego_pose.json", "r") as file:
    ego_poses = json.load(file)

# Find the ego_pose entry for each LiDAR file and extract the translation values
translations = []
for timestamp in lidar_timestamps:
    closest_pose = min(ego_poses, key=lambda x: abs(x["timestamp"] - timestamp))
    translations.append(closest_pose["translation"])

# Convert translations to x, y, z for plotting
x_values = [t[0] for t in translations]
y_values = [t[1] for t in translations]
z_values = [t[2] for t in translations]

# Plot the translations as a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(x_values, y_values, c=z_values, cmap='viridis')
plt.colorbar(label='Z Translation')
plt.xlabel('X Translation')
plt.ylabel('Y Translation')
plt.title('Scatter plot of translations from ego_pose')
plt.grid(True)
plt.show()
