################################################################################################################################################################
#nuscenesのチュートリアル


import matplotlib.pyplot as plt
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

nusc = NuScenes(version='v1.0-mini', dataroot='C://Users/divin/OneDrive - 梅村学園　中京大学/MD/卒業研究/data/nuScenes mini us', verbose=True)

#nusc.list_scenes()# ロードした nuScenes データベース内のシーンのリストを表示するメソッド

my_scene = nusc.scene[1]# 1が使用してるシーン番号
# print(my_scene)

first_sample_token = my_scene['first_sample_token']
# The rendering command below is commented out because it tends to crash in notebooks
# nusc.render_sample(first_sample_token)
my_sample = nusc.get('sample', first_sample_token)#シーン１の最初の画像に関する対応のデータいろいろ
# print(my_sample)

#nusc.list_sample(my_sample['token'])#シーン１の最初の画像に関する対応のデータいろいろとアノテーション

#print(my_sample['data'])#シーン１の最初の画像に関する対応のデータいろいろ

sensor = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
#print(cam_front_data)指定した画像に関するいろいろ

#nusc.render_sample_data(cam_front_data['token'])#アノテーション画像

my_annotation_token = my_sample['anns'][18]
my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)
#print(my_annotation_metadata)#サンプルで見られるオブジェクトの位置を定義するバウンディングボックスを指します。すべての位置データは、グローバル座標系

#nusc.render_annotation(my_annotation_token)#画像出るはずがでない

my_instance = nusc.instance[599]
#print(my_instance)

instance_token = my_instance['token']
#nusc.render_instance(instance_token)

#print("First annotated sample of this instance:")
#nusc.render_annotation(my_instance['first_annotation_token'])

#print("Last annotated sample of this instance")
#nusc.render_annotation(my_instance['last_annotation_token'])

#nusc.list_categories()注釈のオブジェクトの割り当て

#print(nusc.category[9])特定のカテゴリのメタデー

#nusc.list_attributes()属性と、特定の属性に関連付けられたアノテーションの数

my_instance = nusc.instance[27]# 特定のインスタンスのアノテーションを通じて属性がどのように変わるかを調査
first_token = my_instance['first_annotation_token']
last_token = my_instance['last_annotation_token']
nbr_samples = my_instance['nbr_annotations']
current_token = first_token

i = 0
found_change = False
while current_token != last_token:
    current_ann = nusc.get('sample_annotation', current_token)
    current_attr = nusc.get('attribute', current_ann['attribute_tokens'][0])['name']
    
    if i == 0:
        pass
    elif current_attr != last_attr:
        print("Changed from `{}` to `{}` at timestamp {} out of {} annotated timestamps".format(last_attr, current_attr, i, nbr_samples))
        found_change = True

    next_token = current_ann['next']
    current_token = next_token
    last_attr = current_attr
    i += 1


#print(nusc.visibility)特定のアノテーションの可視ピクセルの割合として定義され、それは6つのカメラフィード上でグループ化され、4つのビンに分けられます。

anntoken = 'a7d0722bce164f88adf03ada491ea0ba'
visibility_token = nusc.get('sample_annotation', anntoken)['visibility_token']

#print("Visibility: {}".format(nusc.get('visibility', visibility_token)))
#print(nusc.render_annotation(anntoken))

anntoken = '9f450bf6b7454551bbbc9a4c6e74ef2e'
visibility_token = nusc.get('sample_annotation', anntoken)['visibility_token']

#print("Visibility: {}".format(nusc.get('visibility', visibility_token)))
#print(nusc.render_annotation(anntoken))

#print(nusc.sensor)取得するデータの異なる視点やセンサータイプがわかります。

#print(nusc.sample_data[10])特定のタイムスタンプでのセンサーデータにアクセスしたり、データがどのセンサーから取得されたかを知ることができます。

#print(nusc.calibrated_sensor[0])特定のセンサーのキャリブレーション情報,注: translation（移動）およびrotation（回転）のパラメータは、エゴ車両のボディフレームに対して与えられています

#print(nusc.ego_pose[0])エゴ車両の姿勢に関する情報が格納されています

#print("Number of `logs` in our loaded database: {}".format(len(nusc.log)))データが抽出されたログ情報
#print(nusc.log[0])

#print("There are {} maps masks in the loaded dataset".format(len(nusc.map)))地図情報
#print(nusc.map[0])

#my_sample = nusc.sample[10]
#nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')画像にLiDARのポイントクラウドをプロットし

#nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', render_intensity=True)レーダー

#nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='RADAR_FRONT')サンプルのすべてのサンプルデータにわたるすべてのアノテーションをプロットすることもできます

#my_sample = nusc.sample[20]

# The rendering command below is commented out because it may crash in notebooks
# nusc.render_sample(my_sample['token'])

#nusc.render_sample_data(my_sample['data']['CAM_FRONT'])#バウンディングボックス

#nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True)
#さらに、複数のスイープからのポイントクラウドを集約することで、より密なポイントクラウドを取得することができます。
#nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)

#RadarPointCloud.disable_filters()
#nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)
#RadarPointCloud.default_filters()
#nusc.render_annotation(my_sample['anns'][22])

#my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]#フルシーンをビデオとして描画
# The rendering command below is commented out because it may crash in notebooks
# nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')# The rendering command below is commented out because it may crash in notebooks
# nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')
# The rendering command below is commented out because it may crash in notebooks
# nusc.render_scene(my_scene_token)
#nusc.render_egoposes_on_map(log_location='singapore-onenorth')特定の場所に対する地図上のすべてのシーンを視覚化
