# Used to merge inverted gestures into one dataset, before adding them to the csv file.
# Inverted: hand's palm faces away from camera.

import pandas as pd
import csv
from extract_landmarks import write_json

# total number of samples after merging.
N = 24000
half = N // 2

# seed used for sampling
SEED = 100

# path1 for non-inverted class, path2 for inverted class.
path1 = "../annotations/train/peace.json"
path2 = "../annotations/train/peace_inverted.json"

# paths for temporary storage of both classes
temp_path1 = "./preprocessing/temp/class1.csv"
temp_path2 = "./preprocessing/temp/class2.csv"

# the merged samples will be placed into this dataset.
csv_path = "../model_data/training_data.csv"

# for reference. 'N/A' and 'no_gesture' should not be encountered.
class_id_map = {"fist": 0, "like": 1, "dislike": 2, "one": 3, "middle_finger": 4, "little_finger": 5, "thumb_index": 6,
                "call": 7, "peace": 8, "two_up": 9, "rock": 10, "three_gun": 11, "three2": 12, "three": 13, "three3": 14,
                "ok": 15, "four": 16, "palm": 17, "grip": 18, "stop": 19, "grabbing": 20, "N/A": 21, "no_gesture": 22,
                "peace_inverted": 8, "two_up_inverted": 9, "stop_inverted": 19}

# write both classes into separate files (overwrite pre-existing file)
write_json(path1, temp_path1, True)
write_json(path2, temp_path2, True)

# load and merge N//2 samples from each class
normal = pd.read_csv(temp_path1)
inverted = pd.read_csv(temp_path2)

normal_samples = normal.sample(n=half, random_state=SEED)
inverted_samples = inverted.sample(n=half, random_state=SEED)

merged_samples = pd.concat([normal_samples, inverted_samples])
merged_samples = merged_samples.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle merged dataset

# place merged samples in the target dataset file
merged_samples.to_csv(csv_path, mode="a", header=False, index=False)

print("done")
