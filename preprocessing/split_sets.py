# Split a given csv file's samples into training, test and validation sets
# and append them into destination dataset files.

import pandas as pd
from sklearn.model_selection import train_test_split

# seed used for sampling
SEED = 100

src_path = "./model/keypoint_classifier/keypoint.csv"
train_path = "../model_data/training_data.csv"
test_path = "../model_data/testing_data.csv"
val_path = "../model_data/validation_data.csv"

df = pd.read_csv(src_path)
df.iloc[:, 0] = 21  # replace labels with no_gesture's numeric label

# sample into separate sets, and store them in the correponding paths.
train_df, temp_df = train_test_split(df, test_size=0.25, random_state=SEED)  # 0.75 - training set
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)  # 0.125 - testing, 0.125 - validation

train_df.to_csv(train_path, mode="a", header=False, index=False)
test_df.to_csv(test_path, mode="a", header=False, index=False)
val_df.to_csv(val_path, mode="a", header=False, index=False)
