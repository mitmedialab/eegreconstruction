from pathlib import Path
import shutil
from utils import read_yaml

file_paths = read_yaml("./utils/experiment_easy_paths.yaml")["paths"]
for file_path in file_paths:
    target_dir = Path("./data/images/experiment_subset_easy") / Path(file_path).parent.name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file_path = target_dir / Path(file_path).name
    if target_file_path.exists():
        print(f"File {target_file_path} already exists, skipping.")
    else:
        shutil.copy(file_path, target_dir)