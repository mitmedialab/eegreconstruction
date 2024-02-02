from typing import Dict
import os
import yaml

def write_yaml(path: str, file_name: Dict[str, str]) -> None:
    """
    Writes a yaml file to the specified path.

    Parameters
    ----------
    path : str
        Path to the yaml file.
    file_name : dict of [str, str]
        Dictionary to be written to the yaml file.
    """
    with open(path, 'w') as file:
        yaml.dump(file_name, file)

def read_yaml(path: str) -> Dict[str, str]:
    """
    Reads a yaml file from the specified path.

    Parameters
    ----------
    path : str
        Path to the yaml file.

    Returns
    -------
    dict of [str, str]
        Dictionary read from the yaml file.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)