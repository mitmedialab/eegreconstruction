import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb
import yaml
from pytorch.data_setup.DataModule import DataModule
from pytorch.models.EEGNet import EEGNetv4
from pytorch.models.TSception import TSception
from pytorch.models.EEGChannelNet import ChannelNet
from pytorch.models.EEGNet_Embedding_version import EEGNet_Embedding
from pytorch.models.Conformer import Conformer

def read_config(config_path: str):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def merge_dicts(dict1, dict2):
    """Recursively merge dict2 into dict1."""
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts(dict1[key], dict2[key])
            elif key == "name" and isinstance(dict1[key], str) and isinstance(dict2[key], str):
                dict1[key] += dict2[key]
            else:
                dict1[key] = dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1

def combine_configs(head_config):
    assert "combine" in head_config, "Config is no head config!"
    head_config = head_config.pop("combine")
    for yaml in head_config:
        if "combined_config" in locals():
            combined_config = merge_dicts(combined_config, read_config(yaml))
        else:
            combined_config = read_config(yaml)
    return combined_config

def main(config = None):
    # Initialize a new wandb run
    wandb.init(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize data module
    dm = DataModule(**wandb.config["datamodule"])
    
    # Initialize model
    if wandb.config["model_name"] == "TSCEPTION":
        model = TSception(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 
    elif wandb.config["model_name"] == "EEGNET":
        model = EEGNetv4(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 
    elif wandb.config["model_name"] == "CHANNELNET":
        model = ChannelNet(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"])
    elif wandb.config["model_name"] == "CONFORMER":
        model = Conformer(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"])
    elif wandb.config["model_name"] == "EEGNET_Embedding":
        model = EEGNet_Embedding(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 
    model = model.to(device)

    # Create a ModelCheckpoint callback
    if wandb.config["final_model"] == False:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename="best-model-{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        )
    else:
        print("FINAL MODEL - saving weights only after last epoch")
        checkpoint_callback = ModelCheckpoint(
            monitor=None,
            filename="final-model",
            save_top_k=1,
            save_weights_only=True,
            verbose=True,
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs = wandb.config.trainer["max_epochs"],
        logger = pl.loggers.WandbLogger(save_dir="./results/wandb_logs/"),
        callbacks = [checkpoint_callback, lr_monitor],
        default_root_dir="./results/checkpoints", 
        #pl.callbacks.EarlyStopping(monitor="val_acc")
        accelerator="auto"
    )

    # Train model 
    trainer.fit(model = model, datamodule = dm)

    # if wandb.config["fine_tuning"]:
    #     trainer.fit(model=model)

    # Test model
    #Note: this should only be used once!
    if wandb.config["final_model"] == True:
        trainer.test(datamodule = dm) #test_dataset is integrated in datamodule

    wandb.run.finish()

# Initialize new sweep (Change subject and model name to run different models)
sweep_config = read_config(config_path = "./pytorch/configs/final/P001/EEGNET_P001.yaml")

sweep_config = combine_configs(sweep_config) #Comment out for test run
sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"])

# Run sweep
wandb.agent(sweep_id, function=main)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wandb sweeps for hyperparameter optimization")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="./pytorch/configs/final/P001/EEGNET_P001.yaml",
        help="Path to config file")
    args = parser.parse_args()

    sweep_config = read_config(config_path = args.config_path) # Read config file
    print("Using config: {}".format(args.config_path))
    sweep_config = combine_configs(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"]) # Init sweep
    wandb.agent(sweep_id, function=main) # Run the sweep