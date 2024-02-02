# Utilities
import argparse
import os
import shutil
from utils import read_yaml

# Deep Learning (ResNet50)
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_theme()

# Using CUDA for GPU acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_image_features(image_paths, batch_size = args.batch_size):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting image features"):
        batch_paths = image_paths[i:i+batch_size]
        image_tensors = []
        for image_path in batch_paths:
            img = Image.open(image_path)
            try:
                img_t = transform(img)
            except RuntimeError: # Very few images are 2D and transform fails
                continue
            image_tensors.append(img_t)

        batch_t = torch.stack(image_tensors).to(device)

        with torch.no_grad():
            out = model(batch_t)

        features.extend(out.cpu().numpy())

    return np.array(features)

def _plot_similar_images(best_image_paths, similarity_scores, class_name, n_columns=5):
    output_dir_plot = "./data/visualizations/image_selection_easy/"
    main_image_path = best_image_paths[0]
    n_images = len(best_image_paths)  # account for the main image
    n_rows = n_images // n_columns + (n_images % n_columns > 0)
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(20, 4*n_rows))
    plt.suptitle('Similarity comparison of selected images for class: {}'.format(class_name), fontsize=20)

    # Flatten the axis array if there's more than one row
    if n_rows > 1:
        axs = axs.ravel()

    # Load and display main image
    main_image = Image.open(main_image_path)
    axs[0].imshow(main_image)
    axs[0].set_title("Main Image")
    # Add the red box around the main image
    rect = patches.Rectangle((0, 0), main_image.width-1, main_image.height-1, linewidth=2, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)
    
    # Remove the axis
    axs[0].axis('off')

    # Load and display best matching images
    for i, img_path in enumerate(best_image_paths[1:]):
        img = Image.open(img_path)
        axs[i+1].imshow(img)
        axs[i+1].set_title(f"Similarity: {similarity_scores[i]:.2f}")
        axs[i+1].axis('off')

    # Hide empty subplots
    if n_images < len(axs):
        for i in range(n_images, len(axs)):
            fig.delaxes(axs[i])

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    os.makedirs(output_dir_plot, exist_ok=True)
    plt.savefig(output_dir_plot+"/"+class_name, bbox_inches='tight')

def select_most_similar_images(main_image_path: str, n: int , plot: bool = True):
    """
    Returns the n most similar images to the main image.
    Automatically searches for images in the same directory as the main image.

    Parameters
    ----------
    main_image_path : str
        Path to the main image.
    n : int
        Number of images to return.
    plot : bool
        Whether to plot the most similar and least similar images.

    Returns
    -------
    list
        List of paths to the most similar images.
    """
    other_image_paths = [os.path.dirname(main_image_path)+"/"+file for file in os.listdir(os.path.dirname(main_image_path))]
    class_name = os.path.basename(os.path.dirname(main_image_path))

    print("Extracting features for main image of class {}...".format(class_name))
    main_image_features = _get_image_features([main_image_path])

    print("Extracting features for other images of the class...")
    other_images_features = _get_image_features(other_image_paths)

    print("Calculating similarities...")
    similarities = [cosine_similarity(main_image_features, img_features.reshape(1,-1)).item() for img_features in other_images_features]

    print("Finding most similar images...")
    most_similar_indices = np.argsort(similarities)[-n:][::-1]
    best_images = [other_image_paths[i] for i in most_similar_indices]

    if plot:
        print("Plotting most similar images...")
        _plot_similar_images(best_images, np.array(similarities)[most_similar_indices], class_name)

    # Copy the n_best images for each class to a separate directory
    target_root = "./data/images/experiment_subset_easy"
    print("Copying selection of {} best images to {}.".format(args.n_best, target_root))
    for file_path in best_images:
        target_dir = os.path.join(target_root, class_name)
        os.makedirs(target_dir, exist_ok=True)
        target_file_path = os.path.join(target_dir, os.path.basename(file_path))
        if os.path.exists(target_file_path):
            print(f"File {target_file_path} already exists, skipping.")
        else:
            shutil.copy(file_path, target_dir)

prototypes = read_yaml("./data/prototypical_images_per_class.yaml")
for class_name in prototypes.keys():
    if os.path.exists("./data/images/experiment_subset_easy/{}".format(class_name)):
        print("Class {} already exists - skipping!".format(class_name))
    else:
        select_most_similar_images(prototypes[class_name], n = args.n_best, plot = args.plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subselection of the downloaded images from ImageNet via ResNet50.")
    parser.add_argument("-n", "--n_best", type=int, default=30, help="Number of most similar images to select per image class.")
    parser.add_argument("-p", "--plot", type=bool, default=True, help="Save plot of the n most similar images per class.")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size for image feature extraction.")
    args = parser.parse_args()