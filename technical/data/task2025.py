import os

folder_path = "data/task2025_test"
num_docs = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
print("Number of documents in folder:", num_docs)

import json

json_path = "data/task1_test_labels_2025.json"

with open(json_path, "r") as f:
    data = json.load(f)

num_keys = len(data)
avg_items_per_key = sum(len(v) for v in data.values()) / num_keys

print("Number of keys in JSON:", num_keys)
print("Average number of items per key:", avg_items_per_key)
