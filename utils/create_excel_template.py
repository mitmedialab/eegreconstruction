import os
import pandas as pd
from pathlib import Path

file_dir = "./data/images/experiment_subset_easy"
rows = []
label = 0
for group_label, class_name in enumerate(os.listdir(file_dir)):
    class_dir = file_dir + "/" + class_name
    for file_name in os.listdir(class_dir):
        file_path = class_dir + "/" + file_name
        file_path = ".." + file_path[1:]
        rows.append({'images': file_path, 'group_label': group_label, 'label': label})
        label += 1
df = pd.DataFrame(rows)
df.to_excel("../Experiment/loopTemplate1.xlsx", index=False)