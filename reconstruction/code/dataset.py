
import numpy as np
import os
from einops import rearrange
import torch
from pathlib import Path
import torchvision.transforms as transforms

from typing import Callable
from PIL import Image
import pandas as pd

def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img
    
class EEGDataset():
    """
    EEG: the (preprocessed) EEG files are stored as .npy files (e.g.: ./data/preprocessed/wet/P00x/run_id/data.npy)
    with the individual image labels in the same directory (.../labels.npy) going from 0-599.
    The image labels can be mapped to the image paths via the experiment file used to run Psychopy which is found at
    ./psychopy/loopTemplate1.xlsx

    Images: the images for each image_class are stored in the directory ./data/images/experiment_subset_easy

    Args:
        data_dir: Path to directory containing the data.
            Expects a directory with multiple run-directories. Concatenates all npy files into one dataset.
        image_transform: Optional transform to be applied on images.
        train: Whether to use the training or validation set.
            if True, n-1 runs are returned as dataset
            if False, only the hold-out set is loaded for validation.
        val_run: Name of the run to be used as hold-out set for validation.
            Note: this allows to use a validation set from the same directory or from a different directory by specifying
            a new data_dir for the validation set.
        preload_images: Whether to pre-load all images into memory (speed vs memory trade-off)
    """
    def __init__(
            self, 
            data_dir: Path, 
            image_transform: Callable = None,
            train: bool = True, 
            val_run: str = None,
            preload_images: bool = False,
            ):
        self.preload_images = preload_images
        self.image_paths = np.array(pd.read_excel('./psychopy/loopTemplate1.xlsx', engine='openpyxl').images.values)
        self.image_transform = image_transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if all(p.is_dir() for p in Path(data_dir).iterdir()): #data_dir is a directory with multiple run-directories
            if train:
                self.labels = []
                self.data = []
                for run_dir in data_dir.iterdir():
                    if run_dir.name != val_run:
                        self.labels.append(np.load(run_dir / "labels.npy", allow_pickle=True))
                        self.data.append(np.load(run_dir / "data.npy", allow_pickle=True))
                self.labels = np.concatenate(self.labels)
                self.data = np.concatenate(self.data)
            else:
                #NOTE: REPLACE THIS WHEN EVERYTHING RUNS [::10]
                self.labels = np.load(data_dir / Path(val_run) / Path("labels.npy"), allow_pickle=True)[::10]
                self.data = np.load(data_dir / Path(val_run) / Path("data.npy"), allow_pickle=True)[::10]
        else:
            raise ValueError("data_dir should only contain run-directories")

        self.data = torch.from_numpy(self.data.swapaxes(1,2)).to(self.device) #swap axes to get (n_trials, channels, samples) 
        self.labels = torch.from_numpy(self.labels).long().to(self.device) #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)
        
        if self.preload_images:
            self.images = np.stack([self.image_transform(np.array(Image.open(path[1:]).convert("RGB"))/255.0) for path in self.image_paths])
            self.images = torch.from_numpy(self.images).to(self.device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.preload_images:
            img_idx = self.labels[idx]
            return {'eeg': self.data[idx].float(), 'image': self.images[img_idx]}
        else:
            image_raw = Image.open(self.image_paths[self.labels[idx]][1:]).convert("RGB")
            image = np.array(image_raw) / 255.0
            return {'eeg': self.data[idx].float(), 'image': self.image_transform(image)}