import numpy as np
from pathlib import Path
from typing import List, Literal
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from pytorch.models.EEGNet import EEGNetv4
from pytorch.models.TSception import TSception

TARGET_NAMES = ["Airliner", "Anemone_Fish", "Banana", "Basketball", "Broccoli", "Castle", "Daisy", "Dog", 
                "Face", "Jacko_Lantern", "Orange", "Panda", "Pizza", "Pretzel", "Red_Wine", "School_Bus",
                "Soccer_Ball", "Strawberry", "Tennis_Ball", "Tiger"]

class Dataset():
    """
    Pytorch Dataset Large

    This expects multiple recordings and selects one recording as validation and the rest as training data.

    Args:
        test_path: Path to test dataset.
        label: Whether to use group labels or labels.
    """
    def __init__(
            self, 
            test_path: Path, 
            label: Literal["group", "label"], 
            ):
        if label not in ["group", "label"]:
            raise ValueError("option must be either 'group' or 'label'")
        
        self.test_path = test_path
        self.label_names = "group_labels" if label == "group" else "labels"
        self.data = torch.from_numpy(np.load(self.test_path / "data.npy", allow_pickle=True).swapaxes(1,2)) #swap axes to get (n_trials, channels, samples)
        self.labels = torch.from_numpy(np.load(self.test_path / Path(str(self.label_names) + ".npy"), allow_pickle=True)).long()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def main(
        test_path = "./data/preprocessed/wet/sub-P001_ses-S003_task-Default_run-002_eeg",
        ckpt: str = None
        ):
    dataset = Dataset(test_path = Path(test_path), label = "group")
    test_loader = DataLoader(dataset, batch_size=128)  # Set your appropriate batch_size
    #ckpt = "./results/wandb_logs/EEGNet_dry_classification_runs_1/2smtzmeo/checkpoints/best-model-epoch=118-val_acc=0.28.ckpt"
    model = EEGNetv4.load_from_checkpoint(ckpt)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model.forward(data.float())
            
            # Get the predicted class with the highest score
            _, preds = torch.max(outputs, dim=1)

            # Move labels and preds to CPU for sklearn compatibility
            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Using sklearn's classification report to get per-class metrics
    print(classification_report(all_labels, all_preds, target_names = TARGET_NAMES))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Per class prediction analysis")
    parser.add_argument(
        "--test_path", 
        type=str, 
        default="./data/preprocessed/wet/sub-P001_ses-S003_task-Default_run-002_eeg",
        help="Path to test dataset (default is wet validation)"
        )
    parser.add_argument(
        "--ckpt", 
        type=str, 
        default=None, 
        help="Path to checkpoint (default looks for best checkpoint)"
        )
    args = parser.parse_args()

    if not args.ckpt:
        import glob
        import re
        import os
        ckpt_files = glob.glob('./results/wandb_logs/EEGNet_wet_classification_runs_fixed/**/*val_acc=*.ckpt', recursive=True)
        # Use a generator expression to create tuples of (validation accuracy, filename)
        val_acc_files = ((float(re.search('val_acc=0.(\d{2})', file).group(1)), os.path.normpath(file).replace("\\", "/")) for file in ckpt_files if re.search('val_acc=0.(\d{2})', file))
        # Find the tuple with the highest validation accuracy and get the path
        _, args.ckpt = max(val_acc_files, default=(-1, '')) #max_value_path
        print("Using checkpoint: {}".format(args.ckpt))
    
    main(test_path=args.test_path, ckpt=args.ckpt)
