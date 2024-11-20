import os
import json
import re

data_path = "post_process/data_reorganized"

def src1_sort_key(filename):
    match = re.search(r'_frame_(\d+)_', filename)
    if match:
        return int(match.group(1))  
    return float('inf') 

scene_hierachy = {
    "cam_08": {
        "is_night": True,
        "segments": [(0, 500)]
    },
    "cam_10": {
        "is_night": False,
        "segments": [(0, 500)]
    },
    "src_1": {
        "is_night": False,
        "segments": [(0, 90), (91, 134), (135, 182), (183, 235)]
    },
    "src_2": {
        "is_night": True,
        "segments": [(0, 130), (131, 160)]
    },
}

full_image_path = os.path.join(data_path, "images")
full_yolo_path = os.path.join(data_path, "full_boxes")
prune_yolo_path = os.path.join(data_path, "pruned_boxes")

full_image_list = os.listdir(full_image_path)
full_yolo_list = os.listdir(full_yolo_path)
prune_yolo_list = os.listdir(prune_yolo_path)

# Deals with the src_2_114 gap.
full_image_list.append("src_2_frame_114-_jpg.rf.ad870cae2566f74092b4915ee4048bb0.jpg")
full_yolo_list.append("src_2_frame_114-_jpg.rf.ad870cae2566f74092b4915ee4048bb0.txt")
prune_yolo_list.append("src_2_frame_114-_jpg.rf.ad870cae2566f74092b4915ee4048bb0.txt")

manifest = {
    "cam_08": {},
    "cam_10": {},
    "src_1": {},
    "src_2": {}
}

for scene in scene_hierachy.keys():

    if scene == "src_1":
        scene_image_list = sorted([scene_image_path for scene_image_path in full_image_list if scene_image_path.startswith(scene)], key=src1_sort_key)
        scene_full_yolo_list = sorted([scene_full_yolo_path for scene_full_yolo_path in full_yolo_list if scene_full_yolo_path.startswith(scene)], key=src1_sort_key)
        scene_prune_yolo_list = sorted([scene_prune_yolo_path for scene_prune_yolo_path in prune_yolo_list if scene_prune_yolo_path.startswith(scene)], key=src1_sort_key)
    else:
        scene_image_list = sorted([scene_image_path for scene_image_path in full_image_list if scene_image_path.startswith(scene)])
        scene_full_yolo_list = sorted([scene_full_yolo_path for scene_full_yolo_path in full_yolo_list if scene_full_yolo_path.startswith(scene)])
        scene_prune_yolo_list = sorted([scene_prune_yolo_path for scene_prune_yolo_path in prune_yolo_list if scene_prune_yolo_path.startswith(scene)])

    if scene_hierachy[scene]["is_night"]:
        for seg_no, segment in enumerate(scene_hierachy[scene]["segments"]):
            manifest[scene][f"segment{seg_no * 2}"] = {
                "images": [],
                "full_boxes": [],
                "prune_boxes": []
            }

            for seg_img in scene_image_list[(segment[0])*2:(segment[1]+1)*2:2]:
                manifest[scene][f"segment{seg_no * 2}"]["images"].append(os.path.join(full_image_path, seg_img))

            for seg_yol_ful in scene_full_yolo_list[(segment[0])*2:(segment[1]+1)*2:2]:
                manifest[scene][f"segment{seg_no * 2}"]["full_boxes"].append(os.path.join(full_yolo_path, seg_yol_ful))

            for seg_yol_pru in scene_prune_yolo_list[(segment[0])*2:(segment[1]+1)*2:2]:
                manifest[scene][f"segment{seg_no * 2}"]["prune_boxes"].append(os.path.join(prune_yolo_path, seg_yol_pru))

        
        for seg_no, segment in enumerate(scene_hierachy[scene]["segments"]):
            manifest[scene][f"segment{seg_no * 2 + 1}"] = {
                "images": [],
                "full_boxes": [],
                "prune_boxes": []
            }

            for seg_img in scene_image_list[(segment[0])*2+1:(segment[1]+1)*2+1:2]:
                manifest[scene][f"segment{seg_no * 2 + 1}"]["images"].append(os.path.join(full_image_path, seg_img))

            for seg_yol_ful in scene_full_yolo_list[(segment[0])*2+1:(segment[1]+1)*2+1:2]:
                manifest[scene][f"segment{seg_no * 2 + 1}"]["full_boxes"].append(os.path.join(full_yolo_path, seg_yol_ful))

            for seg_yol_pru in scene_prune_yolo_list[(segment[0])*2+1:(segment[1]+1)*2+1:2]:
                manifest[scene][f"segment{seg_no * 2 + 1}"]["prune_boxes"].append(os.path.join(prune_yolo_path, seg_yol_pru))
    else:
        for seg_no, segment in enumerate(scene_hierachy[scene]["segments"]):
            manifest[scene][f"segment{seg_no}"] = {
                "images": [],
                "full_boxes": [],
                "prune_boxes": []
            }

            for seg_img in scene_image_list[(segment[0]):(segment[1]+1)]:
                manifest[scene][f"segment{seg_no}"]["images"].append(os.path.join(full_image_path, seg_img))

            for seg_yol_ful in scene_full_yolo_list[(segment[0]):(segment[1]+1)]:
                manifest[scene][f"segment{seg_no}"]["full_boxes"].append(os.path.join(full_yolo_path, seg_yol_ful))

            for seg_yol_pru in scene_prune_yolo_list[(segment[0]):(segment[1]+1)]:
                manifest[scene][f"segment{seg_no}"]["prune_boxes"].append(os.path.join(prune_yolo_path, seg_yol_pru))


with open("manifest.json", mode="w+") as manifest_json:
    manifest_json.write(json.dumps(manifest, indent=4))

print("Manifest file created!")