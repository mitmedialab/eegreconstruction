import argparse
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_theme()

parser = argparse.ArgumentParser(description="Subselection of the downloaded images from ImageNet via ResNet50.")
parser.add_argument("-i", "--input_path", type=str, default="./data/images/experiment_subset_easy", help="Path containing the image classes with the respective images in them.")
parser.add_argument("-o", "--output_path", type=str, default="./data/visualizations/image_experiment", help="Path to which the visualizations are saved.")
args = parser.parse_args()

def plot_images(image_dir, out_dir, n_columns=5):
    for class_name in tqdm(os.listdir(image_dir), desc="Plotting images"):
        class_dir = image_dir + "/" + class_name
        image_paths = [class_dir + "/" + file for file in os.listdir(class_dir)]
        n_images = len(image_paths)  # account for the main image
        n_rows = n_images // n_columns + (n_images % n_columns > 0)
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(20, 4*n_rows))
        plt.suptitle('Images of class: {}'.format(class_name), fontsize=30)

        # Flatten the axis array if there's more than one row
        if n_rows > 1:
            axs = axs.ravel()

        # Load and display images
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            axs[i].imshow(img)
            axs[i].axis('off')

        # Hide empty subplots
        if n_images < len(axs):
            for i in range(n_images, len(axs)):
                fig.delaxes(axs[i])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2, top=0.95)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_dir+"/"+class_name, bbox_inches='tight')

if __name__ == "__main__":
    plot_images(image_dir = args.input_path, out_dir = args.output_path)