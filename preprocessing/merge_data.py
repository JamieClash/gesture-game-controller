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
temp_path1 = "../model_data/temp/class1.csv"
temp_path2 = "../model_data/temp/class2.csv"

# the merged samples will be appended onto this dataset.
csv_path = "../model_data/training_data.csv"

# write both classes into separate files (overwrite pre-existing file)
write_json(path1, temp_path1, True)
write_json(path2, temp_path2, True)

# load each file
normal = pd.read_csv(temp_path1)
inverted = pd.read_csv(temp_path2)

# rename headers of both files
header = ["label"] + [f'{i}' for i in range(42)]
normal.columns = header
inverted.columns = header

# sample N//2 samples from each class
normal_samples = normal.sample(n=half, random_state=SEED)
inverted_samples = inverted.sample(n=half, random_state=SEED)

merged_samples = pd.concat([normal_samples, inverted_samples])
print(merged_samples.isna().sum().sum())
merged_samples = merged_samples.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle merged dataset
print(merged_samples.isna().sum().sum())

# place merged samples in the target dataset file
merged_samples.to_csv(csv_path, mode="a", header=False, index=False)

print("done")
