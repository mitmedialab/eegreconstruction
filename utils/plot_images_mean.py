import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Subselection of the downloaded images from ImageNet via ResNet50.")
parser.add_argument("-i", "--input_path", type=str, default="./data/images/experiment_subset_easy", help="Path containing the image classes with the respective images in them.")
parser.add_argument("-o", "--output_path", type=str, default="./data/visualizations/image_experiment_mean", help="Path to which the visualizations are saved.")
args = parser.parse_args()

def plot_mean_image(image_dir, out_dir, n_columns=5):
    os.makedirs(out_dir, exist_ok=True)
    for class_name in os.listdir(image_dir):
        class_dir = image_dir + "/" + class_name
        target_size = (224,224)
        images = [Image.open(class_dir + "/" + file).resize(target_size, Image.ANTIALIAS) for file in os.listdir(class_dir)]
        mean_image = np.mean(np.stack(images, axis = 0), axis=0)
        mean_image_pil = Image.fromarray(np.uint8(mean_image))
        plt.figure(figsize=(10,10))
        plt.title('Mean image of class: {}'.format(class_name), fontsize=25)
        plt.imshow(mean_image_pil)
        plt.axis('off')
        plt.savefig(out_dir+"/"+class_name, bbox_inches='tight')

def plot_multiple_mean_images(image_dir, out_dir, class_list, n_columns=5):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(20, 4))  # Adjust size of figure to accommodate multiple plots
    
    for i, class_name in enumerate(class_list):
        class_dir = image_dir + "/" + class_name
        target_size = (224,224)
        images = [Image.open(class_dir + "/" + file).resize(target_size, Image.ANTIALIAS) for file in os.listdir(class_dir)]
        mean_image = np.mean(np.stack(images, axis = 0), axis=0)
        mean_image_pil = Image.fromarray(np.uint8(mean_image))
        
        plt.subplot(1, n_columns, i+1)  # Set up subplot, incrementing index for each class
        plt.title('{}'.format(class_name.replace("_", " ").capitalize()), fontsize=20)
        plt.imshow(mean_image_pil)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(out_dir+"/"+f"mean_images_{class_list}", bbox_inches='tight')
if __name__ == "__main__":
    #plot_mean_image(image_dir = args.input_path, out_dir = args.output_path)
    class_list = ["airliner", "soccer_ball", "face", "jacko_lantern", "red_wine"]
    plot_multiple_mean_images(image_dir = args.input_path, out_dir = args.output_path, class_list = class_list, n_columns=5)