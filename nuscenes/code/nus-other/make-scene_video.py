#シーンの動画を作る

import cv2
import os

def generate_video_from_images(img_folder, output_file, fps=30):
    # .jpg 形式の画像ファイルのリストを取得
    images = [img for img in os.listdir(img_folder) if img.endswith(".jpg")]
    images.sort()  # ファイル名でソート

    # 画像リストが空でないことを確認
    if not images:
        print("No .jpg images found in the folder.")
        return

    # 最初の画像からフレームサイズを取得
    frame = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, layers = frame.shape

    # ビデオライターを初期化
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 画像を読み込み、動画ファイルに書き込む
    for image in images:
        video_frame = cv2.imread(os.path.join(img_folder, image))
        out.write(video_frame)

    out.release()

# スクリプトを呼び出して動画を生成
img_folder_path = "C:\\Users\\divin\\Downloads\\scene3_sgp\\cam\\cam"  # 画像フォルダのパス
temp_video_path = "temp_video.avi"
output_video_path = "C:\\Users\\divin\\Downloads"

generate_video_from_images(img_folder_path, temp_video_path)

# 一時ファイルを削除
os.remove(temp_video_path)

