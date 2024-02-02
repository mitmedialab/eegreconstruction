import copy
import numpy as np
import os
import pandas as pd
import pathlib
from pathlib import Path
import pyxdf
from typing import Dict, List, Optional, Union
from tqdm.auto import tqdm
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from bad_trial_detection import BadDetection
from preprocessing import Preprocessing

class DatasetLoader():

    def __init__(
            self, 
            dir_path: Path, 
            sessions: List[int], 
            runs: Optional[List[int]],
            fs: int=250,
            event_time: int=2,
            bad_detection_params: Dict = None,
            preprocessing_params: Dict = None,
            #remove_bads: bool = True,
            ):
        """
        Initializes the dataset class.

        The data structure is as follows:
        sven-thesis/
        ├── data/
        │   ├── recordings/
        │   │   ├── subject/ <- dir_path
        │   │   │   ├── session/
        │   │   │   │   ├── eeg/
        │   │   │   │   │   ├── trials

        Attributes
        ----------
        dir_path : Path
            Path to the directory that contains the recordings for a subject.
        sessions : List[int]
            List of session numbers to include in the dataset (e.g.: 1, 2, ...).
            If None, all sessions will be included.
        runs : List[int], optional
            List of run numbers to include in the dataset (e.g.: 1, 2, ...).
        fs : int, optional
            Sampling frequency of the EEG data, by default 250
        event_time : int, optional
            Time in seconds of how long each image was presented, by default 2
        bad_detection_params : Dict, optional
            Parameters for bad trial detection, by default None
        preprocessing_params : Dict, optional
            Parameters for preprocessing, by default None
            
        Note: Choosing more than one session and/or multiple runs for one session will combine multiple trials into one dataset.
        """
        self.dir_path = dir_path if isinstance(dir_path, Path) else Path(dir_path)     
        self.fs = fs
        self.chan_names = ["PO8", "O2", "O1", "PO7", "PO3", "POZ", "PO4", "Pz"]
        self.event_time = event_time
        if sessions:
            if runs:
                assert len(sessions) == 1, "Please specify only one session if you want to select specific runs."
                self.trials = [(self.dir_path / Path(f"ses-S{str(sessions[0]).zfill(3)}") / "eeg" / Path(f"{self.dir_path.stem}_ses-S{str(sessions[0]).zfill(3)}_task-Default_run-{str(i).zfill(3)}_eeg.xdf")) for i in runs]
            else:
                self.trials = [file_path for i in sessions for file_path in (self.dir_path / Path(f"ses-S{str(i).zfill(3)}")).rglob("*.xdf")]
        else: #sessions is None
            self.trials = self.dir_path.rglob("*.xdf") #Use all sessions and runs 
        self.dataset = self.create_dataset()
        self.visualization_path = Path("./data/visualizations")

        #If params for bad detection and/or preprocessing are passed, apply them
        if bad_detection_params:
            print("Applying bad detection...")
            self.mask = self._apply_bad_detection(params = bad_detection_params, preprocessed = False)
            print("...done!")

        if preprocessing_params:
            print("Applying preprocessing...")
            self.dataset = self._apply_preprocessing(params = preprocessing_params)
            print("...done!")
        
    def create_dataset(self, label_bads: bool=True) -> List[pd.DataFrame]:
        """
        Creates a dataset for one trial by combining the EEG and Psychopy data.
        I.e. labels each EEG sample with the corresponding image id and image class id

        Parameters
        ----------
        label_bads : bool, optional
            Whether to label bad EEG channels, by default True

        Returns
        -------
        pd.DataFrame
            Dataframe with EEG data and labels.
        
        """
        dataset = []
        for file_path in tqdm(self.trials, desc="Creating dataset(s)", leave=False):
            data, _ = pyxdf.load_xdf(file_path)

            #Sometimes Lab Recorder accidentally adds another empty stream to the xdf file, so we need to check and get rid of it
            if len(data) > 2:
                for i, _ in enumerate(data):
                    if 0 in data[i]["time_series"].shape:
                        data.pop(i)
                
            eeg, pp = (data[0], data[1]) if data[0]["info"]["name"][0] == "UN-2019.05.50" else (data[1], data[0])
            eeg_time_series = eeg["time_series"] #EEG time series data.
            event_time_series = pp["time_series"] #Event time series data.
            eeg_timestamps = eeg["time_stamps"] #Timestamps for each sample in the EEG time series.
            event_timestamps = pp["time_stamps"] #Timestamps for each sample in the event time series.
            
            # Find the indices of the closest EEG timestamps to each event timestamp
            diff = np.subtract.outer(event_timestamps, eeg_timestamps)
            eeg_indices = np.argmin(np.abs(diff), axis=1)
            start = eeg_indices[0] #index at which first event starts

            # Initialize an array to hold the labels for the EEG samples
            labels = np.empty(eeg_timestamps.shape, dtype=int)
            labels.fill(90) #Fill with 90 (break)
            group_labels = np.copy(labels)

            # Grab the EEG data from first event onwards and turn into dataframe
            eeg_data = eeg_time_series[start:, :8]
            out = pd.DataFrame(eeg_data, columns=self.chan_names)

            # Iterate over the events
            for i in range(len(eeg_indices) - 1):
                # Label all EEG samples + 2s after the event with the event's label, rest is Break (90)
                event_end = int(eeg_indices[i] + self.event_time*self.fs)
                group_labels[eeg_indices[i]:event_end] = int(event_time_series[i][1])
                labels[eeg_indices[i]:event_end] = int(event_time_series[i][2])
            
            # Label the remaining EEG samples with the last event's label
            labels[eeg_indices[-1]:int(eeg_indices[-1]+self.event_time*self.fs)] = event_time_series[-1][1]
            group_labels[eeg_indices[-1]:int(eeg_indices[-1]+self.event_time*self.fs)] = event_time_series[-1][2]
            df = pd.concat([pd.DataFrame({'time': eeg_timestamps[start:], 'label': labels[start:], 'group_label': group_labels[start:]}), out], axis=1)
            df = df[:int(eeg_indices[-1]+self.event_time*self.fs)] #Remove the rest of the EEG samples that don't have a label
            
            #BUGFIX: For some reason, Psychopy sometimes switches group_label with the label for one example. This fixes that.
            incorrect_group_label = df[(df['group_label'] > 19) & (df['group_label'] != 90)]["group_label"].unique()
            labels_per_group = df[df['group_label']<20]['group_label'].value_counts()

            if len(incorrect_group_label) > 0:
                mask = (df['group_label'] == incorrect_group_label[0])
                df.loc[mask, ['group_label', 'label']] = df.loc[mask, ['label', 'group_label']].values

            #In the unfortunate case where the group label switches to an actual group label:
            elif labels_per_group[labels_per_group.idxmax()] > 30*500:
                big_group = labels_per_group.idxmax() # Get the group label with the most instances
                group_max_instances = df[df["group_label"] == big_group] # Select instances in group_max
                label_counts = group_max_instances["label"].value_counts().sort_index() # Get the value counts for the labels within group_max
                for label in label_counts.index: # Find the label which doesn't have an adjacent label
                    if (label - 1 not in label_counts.index) and (label + 1 not in label_counts.index):
                        faulty_label = label
                        mask = (df['group_label'] == big_group) & (df["label"] == faulty_label)
                        df.loc[mask, ['group_label', 'label']] = df.loc[mask, ['label', 'group_label']].values
                        break

            dataset.append(df)
            #Note: Initially, we removed the breaks in between the recordings here, however, that induced high frequency artifacts in the EEG signal.
            #So, at this point, we only remove the "break" after the last recording and leave the rest of the removal to remove_breaks() if necessary.
        return dataset

    def copy(self):
        """
        Returns a copy of the dataset class. Changes to that copy do not affect the original dataset.
        """
        return copy.deepcopy(self)

    def remove_breaks(self):
        """
        Removes the breaks in between the recordings from the dataset.
        Note: Only use after preprocessing (causes high frequency artifacts in filtering)!
        """
        for i in range(len(self.dataset)):
            self.dataset[i] = self.dataset[i][self.dataset[i]['group_label'] < 90]
        return self

    def _apply_bad_detection(self, params, preprocessed):
        """
        Detects bad trials and creates a mask that can be used to remove trials from the dataset.
        """
        BadDetector = BadDetection(data_loader = self, preprocessed = preprocessed)
        return BadDetector(params = params) 
    
    def _apply_preprocessing(self, params):
        """
        Applies preprocessing to the dataset inplace. 
        """
        Preprocessor = Preprocessing()
        return Preprocessor(data = self.dataset, params = params)
    
    def return_dataset(self, 
                       remove_bads = True,
                       save_dir: Path = None,
                       ) -> Union[List, None]: 
        """
        Returns or saves a list of (labels, group_labels, eeg_matrix) for each trial in the dataset.
        The output can either be used for creating the dataset for modeling or for bad trial detection.
        The output for each trial is [600 labels, 600 group labels, 600x500x8 eeg matrix]

        Parameters
        ----------
        save_dir : str, optional
            Path to save the dataset to, by default None.
            If None: Returns the dataset instead of saving.
        
        Returns
        -------
        If save_dir is None:
            list
                List of (labels, group_labels, eeg_matrix) for each trial in the dataset.
        remove_bads : bool, optional
            Whether to remove bad trials from the dataset, by default True
        If save_dir is given:
            None
                Saves the arrays to .npy files at save_dir.

        Note: remove_bads requires to pass bad_detection_params to the class. 
              If remove_bas is False, bads are detected but not removed.
        """
        self.remove_breaks()
        out = []
        for df in self.dataset:
            grouped = df.groupby('label')
            labels = np.array([name for name, group in grouped])
            group_labels = np.array([group['group_label'].iloc[0] for name, group in grouped])
            eeg_matrix = np.array([group[self.chan_names].values for name, group in grouped])

            #Bug Fix: for some reason one recording had a sample with 498 instead of 500 samples
            if eeg_matrix.shape != (600, 500, 8):
                count = 0
                for ix, sample in enumerate(eeg_matrix):
                    if sample.shape != (500, 8):
                        print(f"Sample {ix} has shape {sample.shape}. Excluding...")
                        eeg_matrix = np.delete(eeg_matrix, (ix-count), axis=0)
                        labels = np.delete(labels, (ix-count), axis=0)
                        group_labels = np.delete(group_labels, (ix-count), axis=0)
                        count += 1	
                eeg_matrix = np.stack(eeg_matrix, axis=0)
            out.append((labels, group_labels, eeg_matrix))

        if remove_bads and hasattr(self, "mask"):
            for i in range(len(out)):
                mask = self.mask[i].any(axis=1)
                out[i] = (out[i][0][~mask], out[i][1][~mask], out[i][2][~mask])

        if save_dir:
            for i in range(len(out)):
                file_name = str(self.trials[i].stem)
                for j, name in enumerate(["labels", "group_labels", "data"]):
                    save_path = save_dir / Path(self.trials[i].stem) 
                    save_path.mkdir(parents=True, exist_ok=True)
                    np.save(save_path / name, out[i][j])
        else:
            return out

    def plot_bads(self,):
        """
        Plots the EEG signal for each bad trial on top of the average EEG signal for that image class

        Parameters
        ----------
        trial : int, optional
            Trial number to plot, by default 0
        """
        assert hasattr(self, "mask"), "bad detection must be run before bads can be plotted."
        data = self.return_dataset(remove_bads=False)
        for i in range(len(data)):
            save_path = (self.visualization_path / self.trials[i].stem / "bad_sub_trials")
            save_path.mkdir(parents=True, exist_ok=True)
            labels, group_labels, eeg_matrix = data[i] #get data for trial
            bad_trials = np.argwhere(self.mask[i].any(axis=1)).flatten() #get indices of bad sub_trials for trial
            for idx in bad_trials:
                image_class = group_labels[idx]
                mean_image_class = np.mean(eeg_matrix[np.argwhere(group_labels == image_class).flatten()], axis=0)
                plt.figure(figsize=(20,8*2))
                for k, chan_name in enumerate(self.chan_names):
                    plt.subplot(mean_image_class.shape[1], 1, k+1)
                    plt.plot(mean_image_class[:,k], alpha = 0.5, label = "Mean EEG")
                    plt.plot(eeg_matrix[idx,:,k], label = "Bad EEG")
                    plt.title(chan_name)
                    plt.ylabel("Std. Dev.")
                if k == len(self.chan_names) - 1:
                    plt.xlabel("samples")
                plt.suptitle(f"Recording: {self.trials[i].stem} | Image Class: {image_class} | Label: {labels[idx]}", y=1.02)
                lines_labels = [plt.Line2D([0], [0], color='b', alpha=0.5, lw=2), plt.Line2D([0], [0], color='r', lw=2)]
                plt.figlegend(lines_labels, ['Mean Signal of Image Class', 'EEG of Bad Trial'], loc = 'upper right')
                plt.tight_layout()
                plt.savefig(save_path / Path(f"ImageClass_{image_class}_Label_{labels[idx]}"), bbox_inches='tight')
                plt.close()

    def plot_eeg(self, trial: Optional[int]=0, img_name: str = None):
        """
        Plots the EEG signal for each channel for the whole recording.

        Parameters
        ----------
        trial : int, optional
            Trial number to plot, by default 0
        """
        # Create a new time array representing elapsed minutes since the start of recording
        assert trial < len(self.dataset), f"Trial {trial} does not exist. Please choose a trial between 0 and {len(self.dataset)-1}."
        save_path = (self.visualization_path / self.trials[trial].stem / "full_eeg")
        save_path.mkdir(parents=True, exist_ok=True)
        img_name = Path(img_name) if img_name else Path(f"trial_{trial}")
        df = self.dataset[trial] if trial else self.dataset[0]
        start_time = df['time'].min()
        time = (df['time'] - start_time) / 60

        # Plot the data
        fig, axs = plt.subplots(8, 1, sharex=True, figsize=(15, 10))

        for i, channel in enumerate(self.chan_names):
            axs[i].plot(time, df[channel])
            axs[i].set_ylabel(channel)

        plt.xlabel('Time (minutes)')
        plt.suptitle("EEG signal per channel for whole recording for sub: {self.dir_path.stem} | trial: {trial}")
        plt.tight_layout()
        plt.savefig(save_path / img_name, bbox_inches='tight')

    def plot_eeg_for_image(self, trial: Optional[int]=0, label: int=0, img_name: str = None):
        """
        Plots the EEG signal for each channel for a specific image.

        Parameters
        ----------
        trial : int, optional
            Trial number, by default 0
        label : int, optional
            Image label, by default 0
        """
        assert trial < len(self.dataset), f"Trial {trial} does not exist. Please choose a trial between 0 and {len(self.dataset)-1}."
        save_path = (self.visualization_path / self.trials[trial].stem / Path(f"eeg_for_image_{label}"))
        save_path.mkdir(parents=True, exist_ok=True)
        img_name = Path(img_name) if img_name else Path(f"image_{label}_trial_{trial}")
        df = self.dataset[trial] if trial else self.dataset[0]
        data = df[df['label'] == label]
        start_time = data['time'].min()
        time = (data['time'] - start_time) #/ 250
        # Plot the data
        fig, axs = plt.subplots(8, 1, sharex=True, figsize=(15, 10))
        for i, channel in enumerate(self.chan_names):
            axs[i].plot(time, data[channel])
            axs[i].set_ylabel(channel)

        plt.xlabel('Time')
        plt.suptitle(f"EEG signal per channel for sub: {self.dir_path.stem} | trial: {trial} | image: {label}")
        plt.tight_layout()
        plt.savefig(save_path / img_name, bbox_inches='tight')

    def plot_eeg_for_class(self, trial: Optional[int]=0, group_label: int=0, img_name: str = None):
        """
        Plots the EEG signal for each channel averaged over a specific specific class.
        Note: Averaging over the trials makes only sense with filtering, as electrode drifts are different for each trial.

        Parameters
        ----------
        trial : int, optional
            Trial number, by default 0
        group_label : int, optional
            Image class label, by default 0
        """
        assert trial < len(self.dataset), f"Trial {trial} does not exist. Please choose a trial between 0 and {len(self.dataset)-1}."
        save_path = (self.visualization_path / self.trials[trial].stem / Path(f"mean_eeg_for_class_{group_label}"))
        save_path.mkdir(parents=True, exist_ok=True)
        img_name = Path(img_name) if img_name else Path(f"class_{group_label}_trial_{trial}")
        df = self.dataset[trial] if trial else self.dataset[0]
        group_data = df[df['group_label'] == group_label]
        eeg_per_label = []
        for label in group_data.label.unique():
            eeg_per_label.append(group_data[group_data['label'] == label][self.chan_names])
        mean = np.mean(np.stack(eeg_per_label), axis=0).transpose()

        # Plot the data
        time = np.linspace(0,2.0,500)
        fig, axs = plt.subplots(8, 1, sharex=True, figsize=(15, 10))
        for i, channel in enumerate(self.chan_names):
            axs[i].plot(time, mean[i,:])
            axs[i].set_ylabel(channel, fontsize=16)
            axs[i].set_ylim(-2.2, 2.2)  # Set y-axis limits for each subplot
            axs[i].tick_params(axis='both', which='major', labelsize=14)

        plt.xlabel('Time (s)', fontsize=16)
        plt.xticks(fontsize=14)
        #plt.suptitle(f"Mean EEG signal per channel for sub: {self.dir_path.stem} | run: {trial} | image_class: {group_label}")
        plt.tight_layout()
        plt.savefig(save_path / img_name, bbox_inches='tight')

    def plot_eeg_interactive(self, trial: Optional[int]=0):
        """
        Plot the EEG data for whole recording in an interactive plot using Plotly.
        This allows you to zoom in on specific parts of the recording.

        Parameters
        ----------
        trial : int, optional
            Trial number to plot, by default 0
        """
        # Create a new time array representing elapsed minutes since the start of recording
        assert trial < len(self.dataset), f"Trial {trial} does not exist. Please choose a trial between 0 and {len(self.dataset)-1}."
        df = self.dataset[trial] if trial else self.dataset[0]
        start_time = df['time'].min()
        time = (df['time'] - start_time) / 60

        # Define the plot
        fig = sp.make_subplots(rows=len(self.chan_names), cols=1, shared_xaxes=True, vertical_spacing=0.02)

        # Add traces
        for i, channel in enumerate(self.chan_names, start=1):
            fig.add_trace(go.Scatter(x=time, y=df[channel], name=channel), row=i, col=1)

        # Add title and labels
        fig.update_layout(height=800, width=1400, title_text="EEG signal per channel for whole recording")

        # Remove x-axis title for all but the last subplot
        for i in range(1, len(self.chan_names)):
            fig.update_xaxes(showticklabels=False, row=i)

        # Set x-axis title for the last subplot only
        fig.update_xaxes(title_text="Time (minutes)", row=len(self.chan_names))

        #fig.update_yaxes(title_text="Amplitude", row=len(self.chan_names))
        # Add annotation for y-axis in the middle of the figure
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=-0.06,
            y=0.5,
            text="Amplitude",
            showarrow=False,
            textangle=-90,
            font=dict(size=14)
        )
        # Show the plot
        fig.show()