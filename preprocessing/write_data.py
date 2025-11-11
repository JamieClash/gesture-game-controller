# Used to extract landmark data from json files, and writes (class_id, x1, y1, ..., x21, y21) into target csv file.

from extract_landmarks import write_json

# note that 'peace', 'two_up' and 'stop' have inverse datasets, and need to be dealt with in merge_data.py instead
json_path = "../annotations/train/call.json"
csv_path = "../model_data/training_data.csv"

write_json(json_path, csv_path)
print("done")
