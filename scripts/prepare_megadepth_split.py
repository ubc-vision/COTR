import os
import random
import json


# read the json
valid_list_json_path = './megadepth_valid_list.json'
assert os.path.isfile(valid_list_json_path), 'Change to the valid list json'
with open(valid_list_json_path, 'r') as f:
    all_list = json.load(f)

# build scene - image dictionary
scene_img_dict = {}
for item in all_list:
    if not item[:4] in scene_img_dict:
        scene_img_dict[item[:4]] = []
    scene_img_dict[item[:4]].append(item)

train_split = []
val_split = []
test_split = []
for k in sorted(scene_img_dict.keys()):
    if int(k) == 204:
        val_split += scene_img_dict[k]
    elif int(k) <= 240 and int(k) != 204:
        train_split += scene_img_dict[k]
    else:
        test_split += scene_img_dict[k]

# save split to json
with open('megadepth_train.json', 'w') as outfile:
    json.dump(sorted(train_split), outfile, indent=4)
with open('megadepth_val.json', 'w') as outfile:
    json.dump(sorted(val_split), outfile, indent=4)
with open('megadepth_test.json', 'w') as outfile:
    json.dump(sorted(test_split), outfile, indent=4)
