"""
This script is used to select the best n images per image class from the downloaded ImageNet.
"Best" is defined as the images that are classified with the highest (softmax) probability by ResNet50.
Optional: Save a plot of the best and worst n images per class.
"""

# Utilities
import argparse
import os
from PIL import Image
import shutil
from utils import read_yaml

# Deep Learning (ResNet50)
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.nn import functional as F

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Parse arguments
parser = argparse.ArgumentParser(description="Subselection of the downloaded images from ImageNet via ResNet50.")
parser.add_argument("-n", "--n_best", type=int, default=30, help="Number of best images to select per image class.")
parser.add_argument("-p", "--plot_n", type=int, default=5, help="Save plot of the top and bottom n images per class.")
args = parser.parse_args()

def _read_imagenet_classes(filepath):
    class_dict = {}
    with open(filepath, 'r') as file:
        for index, line in enumerate(file):
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                continue
            class_id, class_label = parts
            class_dict[class_id] = (index, class_label)
    return class_dict

def _find_class_positions(image_ids, imagenet_classes):
    positions = {}
    for class_name, class_id in image_ids.items():
        if class_id in imagenet_classes:
            positions[class_name] = imagenet_classes[class_id][0]
        else:
            print(f"Warning: Class {class_name} with ID {class_id} not found in imagenet_classes.")
    return positions

def _plot_images(best_images, best = True, n_images = 5):
    # Define output path
    output_dir_plot = "./data/visualizations/image_selection_hard/"
    output_path_plot = os.path.join(output_dir_plot, "best_{}_images.png".format(n_images)) if best else os.path.join(output_dir_plot, "worst_{}_images.png".format(n_images))

    # Define the grid
    rows = len(best_images)
    cols = n_images # top 5 images per class

    fig = plt.figure(figsize=(20, 4*rows))

    for i, (class_name, images) in enumerate(best_images.items()):
        # Take top  n_images or lowest n_images
        if best:
            top_images = images[:n_images]
        else:
            top_images = images[-n_images:]
        
        # Add an extra subplot for the class label
        ax_label = fig.add_subplot(rows, cols+1, i*(cols+1) + 1)
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, class_name, rotation='vertical', 
                      horizontalalignment='center', verticalalignment='center', fontsize=12)
        
        for j, ((path, _), accuracy) in enumerate(top_images):
            ax = fig.add_subplot(rows, cols+1, i*(cols+1) + j + 2)
            
            # Load image
            img = Image.open(path)
            # Plot image
            ax.imshow(img)
            ax.axis('off')
            # Add title with accuracy
            ax.set_title(f'Accuracy: {accuracy.item():.4f}')

    # Adjust left space
    fig.subplots_adjust(left=0.05)
    
    # Save the figure if a save path is provided
    os.makedirs(output_dir_plot, exist_ok=True)
    plt.savefig(output_path_plot, bbox_inches='tight')

class ImageNetModel(LightningModule):
    def __init__(self, classes):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()
        self.classes = classes

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits = self.model(x)
            scores = F.softmax(logits, dim=1)
            scores_for_target = torch.gather(scores, 1, self.classes[y].unsqueeze(1))
        return scores_for_target

assert os.path.exists("./data/selected_image_ids.yaml"), "Please run the notebook 'utils/create:image_id_yaml.py' first to generate the file 'data/selected_image_ids.yaml'."
assert os.path.exists("./data/LOC_synset_mapping.txt"), "Missing file 'LOC_synset_mapping.txt' which maps image classes to their respective id."
assert os.path.exists("./data/images/image_subset"), "Missing folder 'data/images/image_subset' which contains the images to be evaluated."
image_ids = read_yaml('./data/selected_image_ids.yaml')
image_folder = "./data/images/image_subset"
imagenet_classes = _read_imagenet_classes("./data/LOC_synset_mapping.txt")
class_indices = _find_class_positions(image_ids, imagenet_classes) #{class_name: loc in softmax output}
target_indices = torch.tensor(list(class_indices.values()), dtype=torch.int64).to("cuda") # [loc in softmax output]

# Prepare the dataset and data loader (transforms are the same as for original resnet50)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading images...")
dataset = datasets.ImageFolder(root=image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

print("Loading model...")
# Initialize the model and trainer
model = ImageNetModel(target_indices)
trainer = Trainer()

print("Running ResNet50...")
# Generate predictions
predictions = trainer.predict(model, dataloader) #n_batches x batch_size
predictions = torch.vstack(predictions) #n_batches*batch_size x 1

print("Selecting images...")
# Group images by class and sort them based on the softmax output
best_images = {}
for image_path, score in zip(dataset.imgs, predictions):
    class_name = os.path.basename(os.path.dirname(image_path[0]))
    if class_name not in best_images:
        best_images[class_name] = []
    best_images[class_name].append((image_path, score))

# Select the n_best images for each class (and the worst n_plot images)
worst_images = {}
for class_name, images in best_images.items():
    images.sort(key=lambda x: x[1], reverse=True)
    best_images[class_name] = images[:args.n_best]
    worst_images[class_name] = images[-args.plot_n:]

# Copy the n_best images for each class to a separate directory
target_root = "./data/images/experiment_subset_hard"
print("Copying selection of {} best images to {}.".format(args.n_best, target_root))
os.makedirs(target_root, exist_ok=True)
for class_name, image_paths in best_images.items():
    for image_path in image_paths:
        target_dir = os.path.join(target_root, class_name)
        os.makedirs(target_dir, exist_ok=True)
        target_file_path = os.path.join(target_dir, os.path.basename(image_path[0][0]))
        if os.path.exists(target_file_path):
            print(f"File {target_file_path} already exists, skipping.")
        else:
            shutil.copy(image_path[0][0], target_dir) # Copy file to target directory

# Plot the images (Optional)
if args.plot_n:
    print("Plotting {} best/worst images...".format(args.plot_n))
    _plot_images(best_images, best=True, n_images=args.plot_n)
    _plot_images(worst_images, best=False, n_images=args.plot_n)