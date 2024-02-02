from pathlib import Path
import numpy as np

# Note: This code allows to choose from multiple pretrained CNNs for feature selection.
# However, in the thesis only the EfficientNet was used as it displayed superior results
# in previous studies.
def data_loader(
        subject: str = "P001",
        val_run: str = "sub-P001_ses-S003_task-Default_run-002_eeg",
        model: str = None,
        fourier: bool = True,
        test: bool = False
        ):
    """
    Load Feature Vectors and labels from directory.

    Args:
        subject: Subject ID.

        val: Name of the validation run.
    
    Returns:
        X: Feature vectors
        y: Labels
    """
    pre_label = "fourier" if fourier else "grayscale"
    if model == "efficientnet":
        file_name = f"{pre_label}_features_efficientnet.npy"
    elif model == "resnet50":
        file_name = f"{pre_label}_features_resnet50.npy"
    elif model == "regnet":
        file_name = f"{pre_label}_features_regnet.npy"
    elif model == "mobilenet":
        file_name = f"{pre_label}_features_mobilenet.npy"
    else:
        raise ValueError("model must be one of efficientnet, resnet50, regnet, mobilenet")
    
    if test:
        train_path = Path(f"./data/preprocessed/wet/{subject}/")
        test_path = Path(f"./data/test_sets/sub_{subject}/wet/")
        X_train = []
        y_train = []
        for run in train_path.iterdir():
            X_train.append(np.load(run / file_name, allow_pickle=True))
            y_train.append(np.load(run / "group_labels.npy", allow_pickle=True))
        for run in test_path.iterdir():
            X_val = np.load(run / file_name, allow_pickle=True)
            y_val = np.load(run / "group_labels.npy", allow_pickle=True)
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        return X_train, y_train, X_val, y_val
    else:
        path = Path(f"./data/preprocessed/wet/{subject}/")
        X_train = []
        y_train = []
        for run in path.iterdir():
            if run.name != val_run:
                X_train.append(np.load(run / file_name, allow_pickle=True))
                y_train.append(np.load(run / "group_labels.npy", allow_pickle=True))
            else:
                X_val = np.load(run / file_name, allow_pickle=True)
                y_val = np.load(run / "group_labels.npy", allow_pickle=True)
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        return X_train, y_train, X_val, y_val

def shuffle(X, y):
    """
    Shuffle X and y in unison.
    """
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="data loading for classical ML")
#     parser.add_argument(
#         "--subject", 
#         type=str, 
#         default="P001", 
#         #default="./pytorch/configs/config_TSception.yaml",
#         help="subject_id"
#         )
#     parser.add_argument(
#         "--val",
#         type=str,
#         default="sub-P001_ses-S003_task-Default_run-002_eeg",
#         help="name of the validation run"
#         )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default=None,
#         help="name of the model"
#         )
#     parser.add_argument(
#         "--fourier",
#         type=bool,
#         default=True,
#         help="whether to use spectrograms or grayscale images"
#         )

#     args = parser.parse_args()
#     data_loader(args.subject, args.val, args.model, args.fourier)