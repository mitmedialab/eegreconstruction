import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb
import yaml
from functools import partial
from pytorch.data_setup.DataModule import DataModule
from pytorch.models.TRANSFER_LEARNING import TransferLearning

def read_config(config_path: str):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def main(config = None):
    # Initialize a new wandb run
    wandb.init(config=config)

    # Initialize data module
    dm = DataModule(**wandb.config["datamodule"])

    # Initialize model
    model = TransferLearning(**wandb.config["model"], epochs = wandb.config.trainer["max_epochs"]) 

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


# Initialize new sweep 
sweep_config = read_config(config_path = "./pytorch/configs/final/P001/transfer_learning_spectrograms.yaml")
sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"])

# Run sweep
wandb.agent(sweep_id, function=main)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wandb sweeps for hyperparameter optimization")
    parser.add_argument(
        "--config_path", 
        type=str,  
        default="./pytorch/configs/final/P001/transfer_learning_spectrograms.yaml",
        help="Path to config file")
    args = parser.parse_args()

    sweep_config = read_config(config_path = args.config_path) # Read config file
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"]) # Init sweep
    wandb.agent(sweep_id, function=main) # Run the sweep