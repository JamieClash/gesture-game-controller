# Used to extract landmark coordinate data from HaGrid annotation json files, 
# then center them with wrist as the center + normalise them with the max norm.
# Writes them into a csv file in the form "class_id, x1, y1, ..., x21, y21" for each row.

import json
import csv
import numpy as np

# for reference. 'N/A' and 'no_gesture' should not be encountered.
class_id_map = {"fist": 0, "like": 1, "dislike": 2, "one": 3, "middle_finger": 4, "little_finger": 5, "thumb_index": 6,
                "call": 7, "peace": 8, "two_up": 9, "rock": 10, "three_gun": 11, "three2": 12, "three": 13, "three3": 14,
                "ok": 15, "four": 16, "palm": 17, "grip": 18, "stop": 19, "grabbing": 20, "N/A": 21, "no_gesture": 22,
                "peace_inverted": 8, "two_up_inverted": 9, "stop_inverted": 19}

# center + noramlise landmark coordinates
def process(landmarks):
    base_x, base_y = landmarks[0]
    centered = [[x - base_x, y - base_y] for x, y in landmarks]

    norms = [np.linalg.norm(coords) for coords in centered]
    max_norm = max(norms) if norms else 1e-8
    normalised = [[x / max_norm, y / max_norm] for x, y in centered]

    flattened = [coord for pair in normalised for coord in pair]
    return flattened

def write_json(json_path, csv_path, write=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    method = "w" if write else "a"

    # write to destination csv
    with open(csv_path, method, newline="") as f:
        writer = csv.writer(f)
        
        for _, entry in data.items():
            if not entry.get("hand_landmarks"):
                continue
            
            # extract gesture label
            index = 0
            if entry.get("labels"):
                if entry["labels"][0] != "no_gesture":
                    label = entry["labels"][0]
                else:
                    label = entry["labels"][1] if len(entry["labels"]) > 1 else "N/A"
                    index = 1
            else:
                label = "N/A"

            id = class_id_map[label]

            # cannot be classified
            if id == 21 or id == 22:
                continue

            # only coordinates for the hand performing the gesture is needed. 
            # it is possible that there are no landmarks available for a specific entry.
            landmarks = entry["hand_landmarks"][index] if entry["hand_landmarks"][index] else []

            if not landmarks:
                continue

            # center by the wrist, scale by the max norm and flatten the 21 pairs of coordinates
            processed_landmarks = process(landmarks)
            
            # [class_id, x1, y1, ..., x21, y21]
            writer.writerow([id] + processed_landmarks)
