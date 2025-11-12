import cv2
import time
from nuscenes.nuscenes import NuScenes

# nuScenesの初期化
dataroot_path = "nuScenes"  # 例
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot_path, verbose=True)

# ユーザが指定したサンプル番号の範囲
start_idx = 39
end_idx = 78

# 表示したいカメラチャネル (CAM_FRONT, CAM_BACK, CAM_FRONT_LEFTなど)
camera_channel = 'CAM_FRONT'

# 表示ウィンドウ名を定義
window_name = "nuScenes Slideshow"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # ウィンドウサイズを自由に変更可能

# 連続表示に使うFPS(1秒あたり何枚表示するか)やインターバル(秒)を設定
display_interval = 0.5  # 1秒ごとに次の画像を表示したい場合

for idx in range(start_idx, end_idx + 1):
    if idx >= len(nusc.sample):
        break  # sample数を超えたら終了
    
    sample = nusc.sample[idx]
    
    # CAM_FRONTのsample_dataトークンを取得
    if camera_channel not in sample['data']:
        print(f"Sample[{idx}] does not have channel: {camera_channel}")
        continue
    
    cam_token = sample['data'][camera_channel]
    sd_rec = nusc.get('sample_data', cam_token)

    # ファイルパス (dataroot + filename) を組み立て
    image_path = f"{nusc.dataroot}/{sd_rec['filename']}"

    # OpenCVで画像を読み込み
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        continue
    
    # 画像を表示
    # ウィンドウが瞬時に閉じないよう、一定時間待機させる
    # キー入力があれば次へ進む or 早送り可能など、使い方に応じて調整
    cv2.imshow(window_name, img)
    
    # ターミナル(コンソール)に情報を出す
    print(f"Showing Sample[{idx}], token={sample['token']}, file={sd_rec['filename']}")
    
    # 指定インターバル待機 (ms単位)
    # ここでは display_interval * 1000 msだけ待ち、待機中にキー押下があればブレイク
    key = cv2.waitKey(int(display_interval * 1000))
    if key == ord('q') or key == 27:  # 'q' or ESCで終了したい場合
        break

cv2.destroyAllWindows()
