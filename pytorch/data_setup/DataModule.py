import numpy as np
import os
from pathlib import Path
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List

from pytorch.data_setup.Dataset import Dataset_Small, Dataset_Large

class DataModule(pl.LightningDataModule):
    """
    The DataModule class allows to built dataset agnostic models as it takes care of all the
    data related stuff. It also allows to easily switch between different datasets.

    Args:
        data_dir: Path to directory containing the data.
            Expects either a directory with multiple run-directories or with .npy files.
            - If multiple sub-directories are found, each sub-directory is considered a run.
            and all npy files are concatenated into one dataset.
            - If no sub-directories are found, a dataset is constructed from the npy files.
        val_run: string specifying the validation run for the large dataset (if None, expects Small Dataset)
        batch_size: Batch size for training and validation.
        num_workers: Number of workers for the dataloader.
        seed: Seed for the stratified random split.

    Example:
    The DataModule can be used to setup the model:
        dm = DataModule(...)
        # Init model from datamodule's attributes
        model = Model(*dm.dims, dm.num_classes)

    The DataModule can then be passed to trainer.fit(model, DataModule) to override model hooks.
    """
    def __init__(
            self, 
            data_dir: str, 
            test_dir: str = None,
            val_run: str = None,
            test_run: str = None,
            batch_size: int = 32, 
            num_workers: int = 0, 
            seed: int = 42, 
            special = None,
            **kwargs):
        
        super().__init__()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.val_run = val_run
        self.test_run = test_run
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.special = special #allows to use grayscale_images or fourier_spectrograms instead of data.npy
       
    def setup(self, stage=None):
        #Loads data in from file and prepares PyTorch tensor datsets for each split.
        #If you don't mind loading all datasets at once, stage=None will load both, train (+val) and test.
        if stage == "fit" or stage is None:
            if self.val_run: #val_run is specified when we train across sessions (Large Dataset)
                self.train_dataset = Dataset_Large(Path(self.data_dir), label = "group", train = True, val_run = self.val_run, special = self.special)
                self.val_dataset = Dataset_Large(Path(self.data_dir), label = "group", train = False, val_run = self.val_run, special = self.special)
            else:
                self.train_dataset = Dataset_Large(Path(self.data_dir), label = "group", train = True, val_run = None, special = self.special)
                #Unfortunately, we need to specify some validation_set for the PL Trainer. Therefore, we just pick the first recording as
                #a dummy validation_set. This leads to a high validation accuracy, but we don't care about the validation accuracy anyway
                #when testing the final model.
                val_run_dummy = os.listdir(self.data_dir)[0]
                self.val_dataset = Dataset_Large(Path(self.data_dir), label = "group", train = False, val_run = val_run_dummy, special = self.special)
                # self.dataset = Dataset_Small(Path(self.data_dir), label = "group", train = True)
                # train_idx, val_idx = self._stratified_random_split(dataset=self.dataset, split=[0.8, 0.2], seed=self.seed)
                # self.train_sampler = SubsetRandomSampler(train_idx)
                # self.val_sampler = SubsetRandomSampler(val_idx)
        if stage == "test" or stage is None:
            if self.test_dir:
                self.test_dataset = Dataset_Large(Path(self.test_dir), label = "group", train = False, val_run = self.test_run, special = self.special)
            else:
                pass

    # def _stratified_random_split(self, dataset, split: List = [0.8, 0.2], seed: int = None):
    #     #Splits a dataset into train and validation set while preserving the class distribution.
    #     #Not used anymore
    #     np.random.seed(seed) if seed else None
    #     train_idx = []
    #     val_idx = []
    #     labels = dataset.labels.numpy()
    #     for label in np.unique(labels):
    #         label_loc = np.argwhere(labels == label).flatten()
    #         np.random.shuffle(label_loc)
    #         n_train = int(split[0]*len(label_loc))
    #         train_idx.append(label_loc[:n_train])
    #         val_idx.append(label_loc[n_train:])
    #     train_idx = np.concatenate(train_idx)
    #     val_idx = np.concatenate(val_idx)
    #     np.random.shuffle(train_idx)
    #     np.random.shuffle(val_idx)
    #     return train_idx, val_idx
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle = True) #, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle = False) #, sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        print("Not implemented yet")
        pass
