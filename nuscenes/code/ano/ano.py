import json
from nuscenes.nuscenes import NuScenes

# 例: nuScenes miniデータセットを読み込む（パスは適宜変更）
nusc = NuScenes(version='v1.0-mini',
                dataroot = "D:\\Users\\wakamatsu.k\\Desktop\\ITS\\nuscenes",
                verbose=True)


# フレーム番号の範囲を指定
start_sample_number = 161
end_sample_number = 201

# 結果を格納するリスト(もしくは辞書)を用意
# ここでは各フレームごとに["sample_token", "annotations"]をまとめて保存する
frames_info = []

for sample_number in range(start_sample_number, end_sample_number + 1):
    # nuScenesのsampleを取得
    my_sample = nusc.sample[sample_number]

    # このフレームのサンプルトークン
    sample_token = my_sample['token']

    # アノテーション(=物体)のトークン一覧
    ann_tokens = my_sample['anns']

    # このフレームに含まれるアノテーション情報を格納するためのリスト
    annotations_info = []

    for ann_token in ann_tokens:
        ann_rec = nusc.get('sample_annotation', ann_token)

        # カテゴリ名（shortcutとして sample_annotation に直接入っている）
        category_name = ann_rec['category_name']  # 例: 'vehicle.car', 'human.pedestrian.adult', etc.

        # 属性 (attribute_tokens のリストを実際の名前に変換)
        attr_names = []
        for attr_token in ann_rec['attribute_tokens']:
            attr_rec = nusc.get('attribute', attr_token)
            attr_names.append(attr_rec['name'])  # 例: "vehicle.moving", "pedestrian.standing" 等

        # 可視率
        visibility_token = ann_rec['visibility_token']  # 文字列 '1' / '2' / '3' / '4'
        visibility_rec = nusc.get('visibility', visibility_token)
        visibility_level = visibility_rec['level']      # 'v0-40', 'v40-60', 'v60-80', 'v80-100'

        # (任意) 3Dバウンディングボックスの中心や大きさを参照したければ以下で
        # translation = ann_rec['translation']  # [x, y, z]
        # wlh = ann_rec['size']                # width, length, height

        annotations_info.append({
            'annotation_token': ann_token,
            'category_name': category_name,
            'attributes': attr_names,
            'visibility': visibility_level,
            # 'translation': translation,
            # 'size': wlh,
        })

    # このフレームの情報をまとめてリストに追加
    frames_info.append({
        'sample_number': sample_number,
        'sample_token': sample_token,
        'annotations': annotations_info
    })

# 取得結果をコンソールに出力
for frame_data in frames_info:
    print(f"--- Frame sample_number: {frame_data['sample_number']} ---")
    print(f"Sample token : {frame_data['sample_token']}")
    print(f"Number of annotations : {len(frame_data['annotations'])}")

    for ann_info in frame_data['annotations']:
        print(f"  - ann_token     : {ann_info['annotation_token']}")
        print(f"    category_name : {ann_info['category_name']}")
        print(f"    attributes    : {ann_info['attributes']}")
        print(f"    visibility    : {ann_info['visibility']}")
    print()

# 必要に応じて jsonファイルなどに保存するなら、以下のようにする
# with open("annotation_summary.json", "w", encoding="utf-8") as f:
#     json.dump(frames_info, f, indent=2, ensure_ascii=False)
