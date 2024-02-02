from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Optional, List, Any, Dict

#from dataset_loader import DatasetLoader
from preprocessing import Preprocessing

class BadDetection:
    def __init__(
        self, 
        data_loader: DatasetLoader,
        fs = 250.0,
        preprocessed = False,
    ):
        """ 
        Class for detecting bad sub-trials in the data.

        Attributes
        ----------
        data_loader : DatasetLoader
            DatasetLoader object containing the data.
        fs : float
            Sampling frequency of the data.
        preprocessed : bool
            Whether the data is already preprocessed or not.
            If False, the data will be filtered before bad detection.
        """
        self.fs = fs
        self.chan_names = data_loader.chan_names
        self.preprocessed = preprocessed
        #Applies filtering if data is unpreprocessed
        if not self.preprocessed:
            self.data = data_loader.copy() #Create copy to not change original data
            Preprocessor = Preprocessing()
            self.data.dataset = Preprocessor.notch_filter(self.data.dataset)
            self.data.dataset = Preprocessor.bandpass_filter(self.data.dataset)
            self.data.remove_breaks() # Remove breaks from the data
            self.data = self.data.return_dataset() #Returns list of (label, group_label, data) & removes breaks
        else:
            if 90 in data_loader.dataset[0]["label"].unique():
                data_loader.remove_breaks() # Remove breaks from the data
            self.data = data_loader.return_dataset() #Returns list of (label, group_label, data) & removes breaks
            
        self.n_sub_trials = len(self.data[0][2])
        self.bad_masks = [np.zeros((len(self.data[i][2]), len(self.chan_names))) for i in range(len(self.data))]

    def __call__(self, params: Dict[str, Dict[str, Any]] = None):
        """
        Applies all bad detection methods to the data.
        Detects sub_trials with:
         - nans
         - flat channels
         - bad correlation patterns

        Parameters
        ----------
        params : Dict[str, Dict[str, Any]]
            Dictionary defining which bad detection methods to apply to the data.
            The keys of the dictionary are the names of the bad detection methods.
            The values of the dictionary are dictionaries containing the parameters for the bad detection methods.
            Example: {"bad_by_nan": {}, "bad_by_flat": {"flat_thresh": 1e-10}, "bad_by_correlation": {"min_threshold": 0.2, "max_threshold": 0.975}}
        
        Returns
        -------
        bad_masks : List[np.ndarray]
            List of bad masks for each trial.
        """
        for func_name, func_params in params.items():
            # Check if method exists
            if hasattr(self, func_name):
                # Get method
                func = getattr(self, func_name)
                # Call method with parameters
                func(**func_params)
            else:
                raise ValueError(f'Unknown function: {func_name}')
            
        # if self.remove:
        #     for i in range(len(self.data)):
        #         mask = self.bad_masks[i].any(axis=1)
        #         self.data[i] = (self.data[i][0][~mask], self.data[i][1][~mask], self.data[i][2][~mask])
        return self.bad_masks

    def bad_by_nan(self) -> BadDetection:
        """
        Detect bad sub-trials by checking for NaN values in the data.

        Updates self.bad_masks with the bad sub-trials by nan.
        """
        for i, trial in enumerate(self.data):
            nan_mask = np.isnan(trial[2]).any(axis=1) #Shows for each sub_trial if there is a nan in any channel
            self.bad_masks[i][nan_mask]=1
        return self

    def bad_by_flat(self, flat_thresh: float = 1e-9) -> BadDetection:
        """
        Detect bad sub-trials by checking for flat channels in the data.
        Flat channels are defined as channels with either a median absolute deviation from the median 
        or a standard deviation below flat_thresh.
        If the data is already preprocessed, only the standard deviation is concidered.

        Updates self.bad_masks with the bad sub-trials by flat.
        """
        for i, trial in enumerate(self.data):  
            mad = (np.median(np.abs(trial[2] - np.median(trial[2], axis=1, keepdims=True)), axis=1, keepdims=True) < flat_thresh).any(axis=1)
            std = (np.std(trial[2], axis=1, keepdims=True) < flat_thresh).any(axis=1)
            self.bad_masks[i][np.logical_or(mad,std)]=1
        return self
    
    def bad_by_correlation(self, min_threshold: float = 0.2, max_threshold: float = 0.99) -> BadDetection:
        """
        Detect channels that a) do not correlate with any other channels or b) completely correlate with another channel. 
        a) indicates that the channel is off compared to the other channels
        b) indicates that a channel is a duplicate of another channel (e.g. due to gel making a connection between two electrodes)

        Updates self.bad_masks with the bad sub-trials by correlation.
        """
        for i, trial in enumerate(self.data): 
            num_trials, num_samples, num_channels = trial[2].shape
            corr_mask = np.zeros((num_trials, num_channels))
            for j in range(num_trials):
                correlation_matrix = np.corrcoef(trial[2][j].T)
                below_threshold = np.max(correlation_matrix, axis=1) < min_threshold # Check if max channel correlation below min threshold
                correlation_matrix[np.diag_indices_from(correlation_matrix)] = 0 # Set diagonal elements to 0
                above_threshold = np.any(correlation_matrix > max_threshold, axis=1) # Check if correlations above max threshold exist
                corr_mask[j] = np.logical_or(above_threshold, below_threshold) # Combine above and below threshold conditions
            self.bad_masks[i][corr_mask.astype(bool)]=1
        return self