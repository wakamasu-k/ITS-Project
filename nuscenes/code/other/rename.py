import os
import sys
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

def get_exif_datetime(filepath):
    """
    画像ファイルの EXIF メタデータから撮影日時を取得する関数。
    撮影日時が存在しない場合は None を返す。
    """
    try:
        with Image.open(filepath) as img:
            exif_data = img._getexif()
            if exif_data is None:
                return None
            
            # EXIF タグから DateTimeOriginal (撮影日時) を探す
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal':
                    return value  # 例: '2022:10:01 12:34:56'
    except Exception as e:
        print(f"EXIF 情報の取得中にエラーが発生: {e}")
    return None

def main():
    """
    指定したフォルダ内にある写真ファイルを撮影日時（EXIF）もしくは更新日時の昇順でソートし、
    第2引数で指定した番号から連番（39,40,41,...など）でリネームするスクリプト。
    """
    if len(sys.argv) < 2:
        print("使い方: python rename_by_date.py <フォルダのパス> [開始番号]")
        print("例: python rename_by_date.py C:/images 39")
        sys.exit(1)
    
    # 対象フォルダのパスを取得
    folder_path = Path(sys.argv[1])
    
    # 開始番号を指定（省略時は 0 にする）
    if len(sys.argv) >= 3:
        try:
            start_idx = int(sys.argv[2])
        except ValueError:
            print("開始番号は整数で指定してください。")
            sys.exit(1)
    else:
        start_idx = 0
    
    # フォルダが存在するか確認
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"指定されたフォルダが存在しません: {folder_path}")
        sys.exit(1)
    
    # 対象とする拡張子（必要に応じて追加/削除）
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # フォルダ内の写真ファイル一覧を取得
    file_list = []
    for ext in exts:
        file_list.extend(folder_path.glob(f'*{ext}'))
    
    # 取得したファイルが存在しない場合
    if not file_list:
        print("フォルダ内に画像ファイルが見つかりません。")
        sys.exit(0)
    
    # ファイルごとに (撮影日時キー, ファイルパス) を持つリストを作成
    data_list = []
    for fpath in file_list:
        exif_dt = get_exif_datetime(fpath)
        if exif_dt is not None:
            # EXIF の撮影日時があれば、そのままでも辞書順で日付順になりやすい
            # ただし正確にソートしたい場合は datetime.strptime でパース推奨
            sort_key = exif_dt.replace(':', '')  # '2022:10:01 12:34:56' -> '20221001 12:34:56'
        else:
            # 撮影日時がない場合はファイルの更新日時を使用
            update_time = fpath.stat().st_mtime  # float (POSIX timestamp)
            sort_key = str(update_time)
        
        data_list.append((sort_key, fpath))
    
    # ソート（撮影日時または更新日時の昇順）
    data_list.sort(key=lambda x: x[0])
    
    # リネームの実行
    # 指定した開始番号からの連番で .jpg に統一したい場合
    current_num = start_idx
    for _, fpath in data_list:
        # 新しいファイル名を作成 例: "39.jpg", "40.jpg", ...
        new_name = f"{current_num}.jpg"
        
        old_path = fpath
        new_path = folder_path / new_name
        
        # 同名ファイルが既に存在する場合は上書きを避ける等の対応が必要
        # ここでは単純に上書きするとして処理
        try:
            old_path.rename(new_path)
            print(f"リネーム: {old_path.name} -> {new_name}")
        except Exception as e:
            print(f"リネームに失敗: {old_path.name} -> {new_name}, エラー: {e}")
        
        current_num += 1

if __name__ == '__main__':
    main()

#使い方
#anaconda promt 
#(cosmo) C:\Users\divin\python(OD)>python number.py "C:\Users\divin\Downloads\match\364_403\camera" 364(振る番号の最初)