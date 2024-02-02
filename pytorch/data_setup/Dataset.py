import numpy as np
from pathlib import Path
from typing import List, Literal
import torch

# Note: We used the Dataset_Small in the beginning to explore models with a few recordings.
# The analyses in the paper are done with Dataset_Large.
class Dataset_Small():
    """
    Pytorch Dataset Small

    This uses the first 25 examples of each image class as training data and the last 5 as validation data.

    Args:
        data_dir: Path to directory containing the data.
            Expects either a directory with multiple run-directories or with .npy files.
            - If multiple sub-directories are found, each sub-directory is considered a run.
            and all npy files are concatenated into one dataset.
            - If no sub-directories are found, a dataset is constructed from the npy files.
        label: Whether to use group labels or labels.
        train: Whether to use the training or test set.
            if True, the last 5 sub_trials for each image_class are removed.
            if False (test), only the last 5 sub_trials for each image_class are used.
    """
    def __init__(self, data_dir: Path, label: Literal["group", "label"], train: bool = True):
        if label not in ["group", "label"]:
            raise ValueError("option must be either 'group' or 'label'")
        
        self.label_names = "group_labels" if label == "group" else "labels"
        self.data = []
        self.labels = []

        if all(p.is_dir() for p in Path(data_dir).iterdir()): #data_dir is a directory with multiple run-directories
            for run_dir in data_dir.iterdir():
                _labels = np.load(run_dir / Path(str(self.label_names) + ".npy"), allow_pickle=True)
                if train:
                    selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[:-5] for label_id in np.unique(_labels)])
                else:
                    selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[-5:] for label_id in np.unique(_labels)])
                
                self.data.append(np.load(run_dir / "data.npy", allow_pickle=True)[selection])
                self.labels.append(_labels[selection])
            self.data = np.concatenate(self.data)
            self.labels = np.concatenate(self.labels)

        elif all(p.suffix == ".npy" for p in Path(data_dir).iterdir()):
            _labels = np.load(data_dir / Path(str(self.label_names) + ".npy"), allow_pickle=True)
            if train:
                selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[:-5] for label_id in np.unique(_labels)])
            else:
                selection = np.concatenate([np.argwhere(_labels == label_id).flatten()[-5:] for label_id in np.unique(_labels)])

            self.data = np.load(data_dir / "data.npy", allow_pickle=True)[selection]
            self.labels = _labels[selection]

        else:
            raise ValueError("data_dir must either contain multiple run-directories or only .npy files")
        
        self.data = torch.from_numpy(self.data.swapaxes(1,2)) #swap axes to get (n_trials, channels, samples) 
        self.labels = torch.from_numpy(self.labels).long() #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class Dataset_Large():
    """
    Pytorch Dataset Large

    This expects multiple recordings and selects one recording as validation and the rest as training data.

    Args:
        data_dir: Path to directory containing the data.
            Expects a directory with multiple run-directories. Concatenates all npy files into one dataset.
        label: Whether to use group labels or labels.
        train: Whether to use the training or test set.
            if True, n-1 runs are returned as dataset
            if False (test), only the hold-out set is loaded for validation.
    """
    def __init__(
            self, 
            data_dir: Path, 
            label: Literal["group", "label"], 
            train: bool = True, 
            val_run: str = None,
            special: str = None
            ):
        if label not in ["group", "label"]:
            raise ValueError("option must be either 'group' or 'label'")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names = "group_labels" if label == "group" else "labels"
        self.data = []
        self.labels = []
        file_name = "data.npy" if special is None else special + ".npy"
        if all(p.is_dir() for p in Path(data_dir).iterdir()): #data_dir is a directory with multiple run-directories
            for run_dir in data_dir.iterdir():
                if run_dir.name != val_run:
                    self.labels.append(np.load(run_dir / Path(str(self.label_names) + ".npy"), allow_pickle=True))
                    self.data.append(np.load(run_dir / file_name, allow_pickle=True))
                else:
                    self.val_labels = np.load(run_dir / Path(str(self.label_names) + ".npy"), allow_pickle=True)
                    self.val = np.load(run_dir / file_name, allow_pickle=True)
      
        else:
            raise ValueError("data_dir should only contain run-directories")

        if train:
            self.labels = np.concatenate(self.labels)
            self.data = np.concatenate(self.data)
        else:
            self.labels = self.val_labels
            self.data = self.val

        if special is None:
            self.data = torch.from_numpy(self.data.swapaxes(1,2)).to(self.device) #swap axes to get (n_trials, channels, samples) 
        self.labels = torch.from_numpy(self.labels).long().to(self.device) #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]