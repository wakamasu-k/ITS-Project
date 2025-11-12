#反射強度の値を見る.1フレーム
import numpy as np
import struct

# LiDARデータのファイルパスを指定
file_path = "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\first\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448744447639.pcd.bin"
# .binファイルから点群データを読み込む
with open(file_path, 'rb') as f:
    content = f.read()
    lidar_data = np.frombuffer(content, dtype=np.float32).reshape(-1, 5)  # ここで5は、x, y, z, intensity, ring indexの5項目を意味します。

# 反射強度情報にアクセス
intensities = lidar_data[:, 3]  # intensityは通常4番目の列に格納されています。

print(intensities)
