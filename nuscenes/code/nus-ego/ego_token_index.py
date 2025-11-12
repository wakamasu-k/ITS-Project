#tokenからindexを抜き出す

import json

# Load the ego_pose.json file
ego_pose_json_path = 'C:\\Users\\divin\\OneDrive - 梅村学園　中京大学\\MD\\卒業研究\\nuScenes\\nuScenes mini us\\v1.0-mini\\ego_pose.json'  # Adjust the path as needed

with open(ego_pose_json_path, 'r') as file:
    ego_poses = json.load(file)

# Load the tokens from token.txt
token_txt_path = 'C:\\Users\\divin\\Downloads\\scene3_sgp\\lidar\\ego_pose_token_list.txt'  # Adjust the path as needed
with open(token_txt_path, 'r') as file:
    tokens = [token.strip() for token in file.readlines()]

# Find and print the index for each token in the ego_poses list
for token in tokens:
    index = next((i for i, ep in enumerate(ego_poses) if ep['token'] == token), None)
    if index is not None:
        #print(f"{index}")#index list作成時に使用
        print(f"The index of the ego_pose with token '{token}' is {index}")
    else:
        print(f"No ego_pose with token '{token}' was found in the file.")
