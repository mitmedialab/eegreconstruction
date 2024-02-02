import requests
from tqdm import tqdm
import tarfile
from utils import read_yaml
from pathlib import Path

def download_file(image_class: str, file_id: str, target_directory: Path):
    # Ensure the target directory exists
    target_directory.mkdir(parents=True, exist_ok=True)

    # Create the URL
    url = f"https://image-net.org/data/winter21_whole/{file_id}.tar"

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the file name from the URL
        file_name = Path(url).name

        # Create the full file path
        file_path = target_directory / file_name

        # Get the total size of the file
        total_size = int(response.headers.get('content-length', 0))

        # Initialize tqdm progress bar
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        # Write the file
        with file_path.open('wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                # Update the progress bar
                progress_bar.update(len(chunk))
                if chunk: # filter out keep-alive new chunks
                    file.write(chunk)

        # Create a subdirectory for the extracted files
        extract_directory = target_directory / image_class
        extract_directory.mkdir(parents=True, exist_ok=True)

        # Extract the tar file
        with tarfile.open(file_path) as tar:
            tar.extractall(path=extract_directory)

        # Remove the tar file
        file_path.unlink()
    else:
        print(f"Failed to download file with ID {file_id} from {url}")

# The selected images and their ids are defined in a yaml file...
image_ids_path = Path("./data/selected_image_ids.yaml")
assert image_ids_path.exists(), "Use create_image_id_yaml.py from utils to create the image id yaml file"
image_ids = read_yaml(image_ids_path)

target_directory = Path("./data/images/image_subset")
progress_bar = tqdm(image_ids.keys()) # Initialize tqdm progress bar for the loop

for image_class in progress_bar:
    if (target_directory / image_class).exists():
        print(f"Skipping image class {image_class} because it already exists")
        continue
    download_file(image_class, image_ids[image_class], target_directory)
    progress_bar.set_description(f"Processed image class: {image_class}, id: {image_ids[image_class]}")
