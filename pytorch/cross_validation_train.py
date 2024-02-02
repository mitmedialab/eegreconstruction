#Cross-Validation
#Note: This is a work-around to do cross-validation with wandb sweeps.

import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb
import yaml
from pytorch.data_setup.DataModule import DataModule
from pytorch.models.EEGNet import EEGNetv4
from pytorch.models.TSception import TSception
from pytorch.models.EEGChannelNet import ChannelNet
from pytorch.models.EEGNet_Embedding_version import EEGNet_Embedding

def read_config(config_path: str):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def main(config = None):
    # Initialize a new wandb run
    wandb.init(config=config)
    
    # Initialize data module
    print("Setting up data module for recording: {}".format(wandb.config["datamodule"]["val_run"]))
    dm = DataModule(**wandb.config["datamodule"])

    # Initialize model
    if wandb.config["model_name"] == "TSCEPTION":
        model = TSception(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 
    elif wandb.config["model_name"] == "EEGNET":
        model = EEGNetv4(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 
    elif wandb.config["model_name"] == "CHANNELNET":
        model = ChannelNet(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"])
    elif wandb.config["model_name"] == "EEGNET_Embedding":
        model = EEGNet_Embedding(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 

    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best-model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs = wandb.config.trainer["max_epochs"],
        logger = pl.loggers.WandbLogger(save_dir="C:/Users/s_gue/Desktop/master_project/sven-thesis/results/wandb_logs/"),
        callbacks = [checkpoint_callback, lr_monitor],
        default_root_dir="C:/Users/s_gue/Desktop/master_project/sven-thesis/results/checkpoints", 
        #pl.callbacks.EarlyStopping(monitor="val_acc")
    )

    # Train model 
    #model.stepsize = np.around(train_set.__len__()*0.8/config["batch_size"]) 
    trainer.fit(model = model, datamodule = dm)
    # Test model
    #trainer.test(datamodule = dm)
    #print("AFTER_TEST")

    wandb.run.finish()

# This is used to iterate over all subject/model combinations
for sub in ["P001", "P002", "P004", "P005", "P006", "P007", "P008", "P009"]:
    sweep_config = read_config(config_path = f"./pytorch/configs/cross_validation/{sub}/EEGNET_{sub}_cv.yaml")

    # Now, we want to iterate over the data_dir and take every recording for val once
    data_dir = sweep_config["parameters"]["datamodule"]["parameters"]["data_dir"]["value"]
    recordings = os.listdir(data_dir)
    for i in range(len(recordings)):
        val_run = recordings[i]
        sweep_config["parameters"]["datamodule"]["parameters"]["val_run"]["value"] = val_run #exchange val_run in config

        #We define the project name by the lr used as that is the main parameter we vary
        sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"] + "_" + str(sweep_config["parameters"]["model"]["parameters"]["lr"]["value"]))

        # Run sweep
        wandb.agent(sweep_id, function=main)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wandb sweeps for hyperparameter optimization")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="./pytorch/configs/cross_validation/P001/EEGNET_P001_cv.yaml",
        help="Path to config file")
    args = parser.parse_args()

    sweep_config = read_config(config_path = args.config_path) # Read config file
    data_dir = wandb.config["datamodule"]["data_dir"]
    recordings = os.listdir(data_dir)
    for i in range(len(recordings)):
        val_run = recordings[i]
        sweep_config["parameters"]["datamodule"]["parameters"]["val_run"]["value"] = val_run #exchange val_run in config

        #We define the project name by the lr used as that is the main parameter we vary
        sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"] + "_" + str(sweep_config["parameters"]["model"]["parameters"]["lr"]["value"]))

        # Run sweep
        wandb.agent(sweep_id, function=main)