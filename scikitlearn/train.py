import numpy as np
import wandb
import yaml
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    confusion_matrix, multilabel_confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from scikitlearn.data_setup.data_setup import data_loader, shuffle

CLASS_NAMES = [
    'airliner',
    'anemone_fish',
    'banana',
    'basketball',
    'broccoli',
    'castle',
    'daisy',
    'dog',
    'face',
    'jacko_lantern',
    'orange',
    'panda',
    'pizza',
    'pretzel',
    'red_wine',
    'school_bus',
    'soccer_ball',
    'strawberry',
    'tennis_ball',
    'tiger'
    ]

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

def calculate_sensitivity_specificity(y_true, y_pred):
    # Get unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    sensitivity_scores = []
    specificity_scores = []

    for i in range(len(classes)):
        cm = confusion_matrix(y_true_bin[:, i], y_pred_bin[:, i])
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
    return sensitivity_scores, specificity_scores

def main(config = None):
    # Initialize a new wandb run
    wandb.init(config=config)
    config = wandb.config

    if config["test"] == True:
        X_train, y_train, X_val, y_val = data_loader(
            subject = config["data"]["subject"], 
            val_run = None, 
            model = config["model_name"], 
            fourier = config["fourier"],
            test = True
            )

    else:  
        X_train, y_train, X_val, y_val = data_loader(
            subject = config["data"]["subject"], 
            val_run = config["data"]["val_run"], 
            model = config["model_name"], 
            fourier = config["fourier"],
            test = False
            )
    
    # Shuffle training data
    X_train, y_train = shuffle(X_train, y_train)

    if config["model"] == "Random_Forest":
        model = RandomForestClassifier(
            criterion = config["criterion"],
            max_depth = config["max_depth"],
            min_samples_leaf = config["min_samples_leaf"],
            min_samples_split = config["min_samples_split"],
            n_estimators = config["n_estimators"]
        )

    elif config["model"] == "SVM":
        model = SVC(
            C = config["C"],
            gamma = config["gamma"],
            kernel = config["kernel"]
        )

    else:
        raise ValueError("model must be Random_Forest or SVM")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Calculate and log metrics
    # Sensitivity and Specificity
    sensitivities, specificities = calculate_sensitivity_specificity(y_val, y_pred)
    sens_data = [[name, sensitivity] for name, sensitivity in zip(CLASS_NAMES, sensitivities)]
    spec_data = [[name, specificity] for name, specificity in zip(CLASS_NAMES, specificities)]
    sens_table = wandb.Table(data=sens_data, columns=["Class", "Sensitivity"])
    spec_table = wandb.Table(data=spec_data, columns=["Class", "Specificity"])

    # Create a bar chart with wandb.plot.bar and log it
    wandb.log({'conf_mat': wandb.plot.confusion_matrix(probs=None,y_true=y_val, preds=y_pred, class_names=CLASS_NAMES)})
    wandb.log({"sensitivity_chart" : wandb.plot.bar(sens_table, "Class", "Sensitivity", title="Class Sensitivities")})
    wandb.log({"specificity_chart" : wandb.plot.bar(spec_table, "Class", "Specificity", title="Class Specificities")})
    wandb.log({'val_acc': accuracy_score(y_val, y_pred)})
    wandb.log({'val_balanced_acc': balanced_accuracy_score(y_val, y_pred)})
    wandb.run.finish()

# Initialize new sweep 
sweep_config = read_config(config_path = f"./scikitlearn/configs/test/P009_EfficientNet_SVM_test.yaml")
#sweep_config = combine_configs(sweep_config) #Uncomment if you are running sweeps on validation as a different config is used that requires combining sub-configs
sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"])
# # Run sweep
wandb.agent(sweep_id, function=main)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wandb sweeps for hyperparameter optimization")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="./scikitlearn/configs/test/P009_EfficientNet_SVM_test.yaml", 
        help="Path to config file")
    args = parser.parse_args()

    sweep_config = read_config(config_path = "./scikitlearn/configs/test/P009_EfficientNet_SVM_test.yaml") # Read config file
    sweep_config = combine_configs(sweep_config) # Combine configs
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["name"]) # Init sweep
    wandb.agent(sweep_id, function=main) # Run the sweep