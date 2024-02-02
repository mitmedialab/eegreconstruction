# Image Classification and Reconstruction from Low-Density EEG

## Starting Guide
1. Clone the repo to your local machine (git clone ...)
2. Use the conda_env.yaml file to create an env for this project (conda env create -f conda_env.yaml) and activate it (conda activate vision)
3. Install the [pytorch version](https://pytorch.org/get-started/locally/) that suits your system
4. From inside the local repo:
  - run "pip install -e ." 

## To reproduce the whole experiment (skip to step 13./14. if you only want to verify the classification results or do reconstructions)
5. Get the images
  - run "python ./utils/download_imagenet_subset.py" to get the ImageNet dataset
  - Download the Face images from [Kaggle](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) (ImageNet does not contain pure face images), unpack it, and put the images into a folder called "face" inside ./data/images/image_subset
  - run "python ./utils/create_experiment_data.py" which will create the actual selection of images for the experiment in ./data/images/image_subset_easy
    This includes 20 image classes with 30 images each.

6. Psychopy setup:
  - Download and install [Psychopy](https://www.psychopy.org/download.html)
  - Open the Psychopy Coder and open the experiment.py from the psychopy folder
    
    Note: the loopTemplate1.xlsx tells the experiment from which images to randomly sample in the loop.
  - Hit "Run Experiment" which will open the experiment scream and start the stream to Lab Streaming Layer.
  - Latency: the timings in Psychopy are based on the screen refresh rate of the monitor in use. Modern displays often post-process images (e.g. in movie or "game mode") which causes a delay. Switch those modes off (for more, see [here](https://psychopy.org/general/timing/millisecondPrecision.html)).

7. EEG Computer Setup:
  - Install Unicorn Suite Hybrid Black software which is the software through which you connect your Gtec Unicorn Hybrid Black headset.
  - Disable the internal Bluetooth adapter of your computer (device manager – Bluetooth – disable)
  - Insert Gtec USB bluetooth dongle.
  - Disable power saving mode (Open the Bluetooth properties of the “Generic Bluetooth Radio” in the device manager. Go to “Power Management” tab and deselect “Allow the computer to turn off this device to save power”.)
  - Open the Unicorn Suite and go to “My Unicorn” tab to connect the headset via bluetooth. If the device blinks slowly, the connection is established.

8. EEG Physical Setup:
  Previous research has shown that the most predictive channel locations for the detection of observed image classes are all located over the occipital and parieto-occipital lobes. Therefore, we have adjusted the default channel locations of the Gtec Unicorn headset. The placements can be taken from the images below.The numbers indicate the channel number (you can find them on the electrodes of the headset).

  ![EEG_order](https://github.com/AttentivU/sven-thesis/assets/129537044/0f4ddcf3-ce9c-4d28-97dc-a094ab65a3eb)
  ![Extended_10_20 svg](https://github.com/AttentivU/sven-thesis/assets/129537044/583155c6-2c35-4888-a4d4-75da702c4352)
    
9. Lab Streaming Layer (LSL):
 
  LSL is used to synchronously measure the EEG activity and the timestamps coming from the experiment in Psychopy.
  For the EEG:
   - Follow this [tutorial](https://www.youtube.com/watch?v=l18lJ7MGU38) to setup the Unicorn LSL.
   - If your device is connected, you can open and start the datastream from the EEG (the device LED will be continuously on if succesful). 
   For Psychopy:
   - Open the Psychopy Coder and start the experiment by hitting the run button.
   ![psychopy_start_run](https://github.com/AttentivU/sven-thesis/assets/129537044/bb9a5e1b-65dc-47a8-b0b7-0a270166a2aa)

  This will open another stream coming from Psychopy. Wait for the popup that asks for the participant and session, but don't click on "Okay", yet. 
  ![psychopy_popup](https://github.com/AttentivU/sven-thesis/assets/129537044/6d30106f-8ee3-477a-9e51-1a810c2c870d)

  At this point, the stream becomes visible to the Lab Recorder. Once both streams are incoming, open the LabRecorder (also from LSL and can be downloaded [here](https://github.com/labstreaminglayer/App-LabRecorder/releases)) and select the open streams to record from them. Select "./data/recordings" as the output folder and let it save the data in BIDS format. Then "start the LabRecorder.
  ![LabRecorder](https://github.com/AttentivU/sven-thesis/assets/129537044/0f794d00-1065-4562-9ee4-b9211d8b01ab)
   
  Now, click "Okay" on the Psychopy Coder popup to start with the experiment. Once the experiment is done, you can stop the LabRecorder. The LabRecorder should have stored a recording file including the EEG and Psychopy stream in the selected folder.

10. Create a clean dataset from recordings (DatasetLoader)
  
  Once we have made a recording and the data is stored under ".../sven-thesis/data/recordings", we can create the dataset. There are 2 different datasets that can be created. One does a bad trial rejection and preprocessing and saves the data as .npy files. The second type of dataset is used for Pytorch and actually creates the dataset suitable for Pytorch from the saved .npy files (see 9.). 

  To create a clean (.npy) dataset, we apply the following steps (default; feel free to change or extend things):

  a. Bad Trial Rejection
    1. Applies notch and bandpass filter if data is raw
    2. Removes the breaks from the data
    3. Cuts the data into trials
    4. Checks for each trial (and channel) if it contains nans, flat areas, or if a channel has very low or high correlations with other channels (e.g. high noise or electrode drift)
    5. Output a mask that can be used to filter out bad trials.

  b. Preprocessing
    1. Notch Filter (60Hz)
    2. Bandpass Filter with a Kaiser Window (1-95Hz)
    3. Normalization (z-scores)
    4. Clamping (>20 |std devs|)
  Alternatively, denoising may be done via wavelet transform

  Note: The steps during the Bad Trial Rejection and Preprocessing can be varied by passing parameter dictionaries to the DatasetLoader class. Check ./dataset/load_data.ipynb for an example of how to run this.

  After the preprocessing, you may either inspect the data with one of the plotting functions or return it. Passing a save_dir to the return_dataset function of the DatasetLoader will save the cleaned data.

11. Classification (./pytorch)

  Once you have the clean dataset(s), you can create a Pytorch Lightning (PL) DataModule from it ("./pytorch/data_setup/"). For that, pass a data directory to the DataModule(data_dir, ...) class. For replicability, you can specify a seed that is used for the dataloader. The data_dir should either be the recordings of one subject (e.g. "./data/preprocessed/sub-PXXX") or a run containing the .npy files for that run (e.g. "./data/preprocessed/sub-PXXX/ses-SXXX/eeg"). In the first case, the .npy files are concatenated into one dataset. For the latter case, just the data from that run is used. The analyses in this study have been conducted across multiple recordings. In order to avoid accidental training on the test-set, we have put the test recordings into a different directory for the study (e.g. "./data/test_set)

  The DataModule can be passed to a PL Trainer. The only other thing it needs is a model, which can be found in "./pytorch/models". The parameters for the model are defined in a config file (see "./pytorch/configs" for config files for different models and subjects) which can be passed to ./pytorch/train.py to do a hyperparameter search using Weights and Biases Sweeps.
  
  Cross-validation: Whereas train.py is used to run the hyperparameter search on a hold-out validation set, the cross_validation_train.py is used to obtain an estimate of how well the model would generalize. Therefore, the model with the optimal hyperparameters is trained on k-1 (preprocessed) recordings and evaluated on the respective hold-out recording.

  Testing: The test dataset should only be used once. Therefore, we have put one recording for each subject into the ./data/test_set directory. To run a model on the test set, a config needs to set the final_model parameter to True and specify the test_directory. 

12. Reconstruction (./reconstruction)

  You may download the pretrained LDM that we use from the Ommer-Lab [here](https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt).
  After downloading, save the model.ckpt in "./reconstruction/pretrains/ldm/label2img".
  The modified EEGNet used to obtain EEG embeddings is already in "./reconstruction/pretrains/EEG"
  You may finetune the LDM with the EEG-Image pairs using the stageB_ldm_finetune script (similar to MindVis)

## How to replicate the classification results
13. The ckpt files of each model for every subject are in "./pytorch/final_classification_ckpts".
  Additionally, you may use the notebook "run_all_test_classification.ipynb" to run the classifiers for each subject.

## How to replicate the reconstruction results
14. Download the [finetuned.pth file](https://drive.google.com/file/d/17XudAyPvN2yWlPmLUH7Ht_rvqyDzJRp6/view?usp=sharing) and put the finetuned LDM file into the "./reconstruction/pretrains/EEG" folder.
  Run the "./reconstruction/code/gen_eval.py" script.


## Repo Overview
Description of the important directories
```

/data (not included in repo)
┣ 📂 images
┃   ┣ 📂 experiment_subset_easy
┃   ┃   ┣ 📂 image_class_xyz: contains 30 images for each image class
┣ 📂 preprocessed
┃   ┣ 📂 dry: dry recordings (only subject 1)
┃   ┣ 📂 wet: wet recordings
┃   ┃   ┣ 📂 sub-id
┃   ┃   ┃   ┣ 📂 sub-id_ses-id_task-Default_run-id_eeg
┃   ┃   ┃   ┃   ┗ 📜 data.npy: preprocessed EEG data
┃   ┃   ┃   ┃   ┗ 📜 grayscale_features_efficientnet.npy: 8-bit grayscale transformed EEG
┃   ┃   ┃   ┃   ┗ 📜 group_labels.npy: class label
┣ 📂 test_sets: contains test recordings for each subject
┃   ┣ 📂 sub-id
┃   ┃   ┣ 📂 dry (only subject 1)
┃   ┃   ┣ 📂 wet
┃   ┃   ┃   ┣ 📂 sub-id_ses-id_task-Default_run-id_eeg
┃   ┃   ┃   ┃   ┗ 📜 data.npy: preprocessed EEG data
┃   ┃   ┃   ┃   ┗ 📜 group_labels.npy: class label

/dataset: initial data loading and preprocessing 
┗ 📜 bad_trial_rejection.py
┗ 📜 dataset_loader.py
┗ 📜 load_data.py
┗ 📜 preprocessing.py
┗ 📜 transformation.py: EEG-to-image transformations

/psychopy
┗ 📜 experiment.py: script to run experiment
┗ 📜 loopTemplate1.xlsx: used to load and label data during experiment
┗ 📜 loopTemplate1_zero_shot.xlsx: for image classes not used in model fitting

/pytorch
┣ 📂 configs: contains configs for every subject & model combination
┣ 📂 models: contains model implementations
┣ 📂 data_setup: Dataset and data loading 
┗ 📜 cross_validation_train.py: training script for CV
┗ 📜 train.py: training script to train/evaluate models
┗ ... rest is legacy of things we tried out

/reconstruction: adapted from [MindVis](https://github.com/zjc062/mind-vis)
┣ 📂 code
┃   ┣ 📂 dc_ldm
┃   ┃   ┣ 📂 models
┃   ┃   ┃   ┗ (adopted from LDM)
┃   ┃   ┣ 📂 modules
┃   ┃   ┃   ┗ (adopted from LDM)
┃   ┃   ┗ 📜 ldm_for_eeg.py (adapted from MindVis)
┃   ┃   ┗ 📜 util.py (adopted from MindVis)

┃   ┗ 📜 config.py: configurations for the main scripts (adapted from MindVis)
┃   ┗ 📜 dataset.py: used to load datasets (adapted from MindVis)
┃   ┗ 📜 eval_metrics.py: evaluation metrics (adopted from MindVis)
┃   ┗ 📜 gen_eval.py: generation of decoded images (adopted from MindVis)
┃   ┗ 📜 stageB_ldm_finetune.py: script to finetune LDM (adapted from MindVis)

┣ 📂 pretrains (not included in repo)
┃   ┣ 📂 EEG
┃   ┃   ┗ 📜 P001_model_config.yaml: config file of EEG encoder
┃   ┃   ┗ 📜 final-model-P001.ckpt: EEG encoder
┃   ┣ 📂 EEG
┃   ┃   ┗ (adopted from MindVis)

/scikitlearn: EEG-to-image approach stuff
┣ 📂 configs: contains configs for every subject & model combination
┣ 📂 data_setup
┃   ┗ 📜 data_setup.py: data loader for 
┗ 📜 train.py: training script to train/evaluate models

/utils: scripts used to download & select images, prepare experiment scripts, and visualize stuff 
```

## Data Availability
The recordings (raw or preprocessed) will be made available upon reasonable request.

## Acknowledgment
This repository profited a lot from other peoples' work. We want to thank [CompVis](https://github.com/CompVis/latent-diffusion) for their pretrained Latent Diffusion Model and [MindVis](https://github.com/zjc062/mind-vis) for their implementation on using fMRI images to condition the LDM. For the reconstruction part we adopted large parts of their code and replaced the fMRI encoder with our EEG encoder.

Additionally, we want to thank several groups for giving open access to the following models which we adapted and used for classification:
- [EEGNet](https://github.com/braindecode/braindecode/tree/master/braindecode)
- [TSCeption](https://github.com/deepBrains/TSception)
- [Conformer](https://github.com/eeyhsong/EEG-Conformer/tree/main)
- [EEGChannelNet](https://github.com/perceivelab/eeg_visual_classification/blob/main/models/EEGChannelNet.py)

## Authors
Project Lead: Nataliya Kosmyna, Ph.D [Email](nkosmyna@mit.edu)

Lead developer and main contributor: Sven Guenther 
[Email](sven.guenther@tum.de)

# TO-DO
- add paper link

# Data + Model
- to access the EEG data please feel out this form
- to access the model please feel out this form

# Copyright
Copyright (C) Massachusetts Institute of Technology - Media Lab 2023-2024 - All Rights Reserved

Unauthorized copying of these files, via any medium is strictly prohibited
This repository cannot be copied and/or distributed without the express permission of Nataliya Kosmyna and Sven Guenther: nkosmyna@mit.edu and sven.guenther@tum.de
