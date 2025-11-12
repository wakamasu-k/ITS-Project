import numpy as np
import cv2
import matplotlib.pyplot as plt

# Adjust this size to change the point size in the overlay
point_size = 1

# LiDAR data loading (including reflectivity)
pointcloud = np.fromfile("C:\\Users\\divin\\Downloads\\kitti\\0000000150.bin", dtype=np.float32).reshape(-1, 4)
pointcloud = pointcloud[pointcloud[:, 0] >= 0, :]

# Loading camera image
image = cv2.imread("C:\\Users\\divin\\Downloads\\kitti\\0000000150.png")

# Setting up camera matrix and transformation matrix
camera_matrix = np.array((721.5377, 0, 609.5593, 0, 721.5377, 172.854, 0, 0, 1), dtype="float").reshape(3, 3)
rotation = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02], dtype="float").reshape(3, 3)
translation = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01], dtype="float").reshape(1, 3)
transform = np.vstack((np.hstack((rotation, translation.T)), np.array([0, 0, 0, 1])))

# Transforming point cloud to camera coordinate system
pointcloud_hom = np.hstack((pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))))
pointcloud_camera = np.dot(transform, pointcloud_hom.T).T
pointcloud_camera = pointcloud_camera[pointcloud_camera[:, 2] >= 0, :3]

# Projecting to image coordinate system
points_projected = np.dot(camera_matrix, pointcloud_camera.T).T
points_projected /= points_projected[:, 2][:, np.newaxis]

# Convert to integer for pixel coordinates
points_projected = points_projected[:, :2].astype(int)

# Color coding for reflectivity
intensities = pointcloud[:, 3]
colors = plt.cm.jet((intensities - intensities.min()) / (intensities.max() - intensities.min()))

# Overlaying points on the image
for i, p in enumerate(points_projected):
    if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
        cv2.circle(image, (p[0], p[1]), point_size, (int(colors[i, 0]*255), int(colors[i, 1]*255), int(colors[i, 2]*255)), -1)

# 画像表示のためのウィンドウを作成し、画像を表示
cv2.imshow('Overlay Image', image)
cv2.waitKey(0)  # キー入力を待つ
cv2.destroyAllWindows()  # ウィンドウを閉じる

# 画像をファイルに保存
#cv2.imwrite('overlay_image.jpg', image)

# 描画する点のサイズを設定（例：3）
depth_point_size = 3

# デプス画像用の背景画像を作成（黒背景）
depth_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

# 投影されたポイントをループし、デプス画像を更新
for i, p in enumerate(points_projected):
    if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
        depth_intensity = int((pointcloud_camera[i, 2] - pointcloud_camera[:, 2].min()) / (pointcloud_camera[:, 2].max() - pointcloud_camera[:, 2].min()) * 255)
        # 点のサイズを変更
        cv2.circle(depth_image, (p[0], p[1]), depth_point_size, depth_intensity, -1)

# デプス画像の表示と保存
cv2.imshow('Depth Image', depth_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# デプス画像をファイルに保存
#cv2.imwrite('depth_image.jpg', depth_image)







































'''
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448744447639.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448745047596.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448745547460.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448746047898.pcd.bin",  
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448746548329.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448747047643.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448747547529.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448748047387.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448748547277.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448749047731.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448749547584.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448750047463.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448750547877.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448751048300.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448751548178.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448752048628.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448752549040.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448753048377.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448753547684.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448754047572.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448754547448.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448755047865.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448755548316.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448756050426.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448756547528.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448757048500.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448757548354.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448758048230.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448758547578.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448759047433.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448759546758.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448760047738.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448760548179.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448761048087.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448761548500.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448762048350.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448762547686.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448763047541.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448763547987.pcd.bin",   
   "C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\pcdbin\\n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448764047314.pcd.bin",
   '''

'''
   nusc.ego_pose[16833],
   nusc.ego_pose[16844],
   nusc.ego_pose[16838],
   nusc.ego_pose[16866],
   nusc.ego_pose[16789],
   nusc.ego_pose[16884],
   nusc.ego_pose[16877],
   nusc.ego_pose[16906],
   nusc.ego_pose[16828],
   nusc.ego_pose[16928],
   nusc.ego_pose[16915],
   nusc.ego_pose[16946],
   nusc.ego_pose[16865],
   nusc.ego_pose[16968],
   nusc.ego_pose[16956],
   nusc.ego_pose[16989],
   nusc.ego_pose[16904],
   nusc.ego_pose[17009],
   nusc.ego_pose[16995],
   nusc.ego_pose[17030],
   nusc.ego_pose[16941],
   nusc.ego_pose[17049],
   nusc.ego_pose[17034],
   nusc.ego_pose[17070],
   nusc.ego_pose[16978],
   nusc.ego_pose[17091],
   nusc.ego_pose[17071],
   nusc.ego_pose[17109],
   nusc.ego_pose[17016],
   nusc.ego_pose[17128],
   nusc.ego_pose[17134],
   nusc.ego_pose[17146],
   nusc.ego_pose[17054],
   nusc.ego_pose[17169],
   nusc.ego_pose[17174],
   nusc.ego_pose[17105],
   nusc.ego_pose[17125],
   nusc.ego_pose[17240],
   nusc.ego_pose[17280],
   nusc.ego_pose[17141],
'''