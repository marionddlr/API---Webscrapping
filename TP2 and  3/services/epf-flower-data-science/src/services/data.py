from fastapi import HTTPException
import json
import os
import pandas as pd

def get_dataset(dataset: str):

    """Retrieve dataset details from the configuration file.

    Args:
        dataset (str): The name of the dataset to fetch.

    Raises:
        HTTPException: Raised if the dataset is not found in the configuration file.
        HTTPException: Raised if the configuration file is not found in the expected directory.

    Returns:
        dict: A dictionary containing the dataset's details (e.g., name and URL).
    """

    with open('TP2 and  3/services/epf-flower-data-science/src/config/config.json', 'r') as f:
        config = json.load(f)

    config_path = 'TP2 and  3/services/epf-flower-data-science/src/config/config.json'

    if dataset not in config:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found.")
    
    if not os.path.exists(config_path):
        raise HTTPException(status_code=503, detail="Configuration file not found in the expected directory.")

    return config[dataset]


def add_dataset(dataset_name: str, dataset_url: str):

    """Add a new dataset to the configuration file.

    Args:
        dataset_name (str): The name of the dataset to add.
        dataset_url (str): The URL of the dataset to add.

    Raises:
        ValueError: Raised if the dataset already exists in the configuration file.

    Returns:
        dict: A dictionary containing the added dataset's details.
    """

    config_path = 'TP2 and  3/services/epf-flower-data-science/src/config/config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    if dataset_name in config:
        raise ValueError(f"Dataset {dataset_name} already exists in the config.")

    config[dataset_name] = {
        "name": dataset_name,
        "url": dataset_url
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return config[dataset_name]


def modify_dataset(dataset_name: str, new_url: str):

    """Modify the URL of an existing dataset in the configuration file.

    Args:
        dataset_name (str): The name of the dataset to modify.
        new_url (str): The new URL to assign to the dataset.

    Raises:
        ValueError: Raised if the dataset does not exist in the configuration file.

    Returns:
        dict: A dictionary containing the updated dataset's details.
    """

    config_path = 'TP2 and  3/services/epf-flower-data-science/src/config/config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    if dataset_name not in config:
        raise ValueError(f"Dataset {dataset_name} does not exist in the config.")

    config[dataset_name]["url"] = new_url

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return config[dataset_name]


def load_dataset(dataset_name: str):

    """
    Load a dataset file and return its content as a dataframe pandas.

    Args:
        dataset_name (str): The name of the dataset to load.

    Raises:
        HTTPException: If the dataset file is not found or cannot be loaded.

    Returns:
        list: The dataset content in a dataframe pandas.
    """

    config_path = 'TP2 and  3/services/epf-flower-data-science/src/config/config.json'

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Configuration file not found at {config_path}."
        )

    if dataset_name not in config:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_name} is not listed in the configuration file."
        )

    dataset_filename = f"{dataset_name}.csv"
    dataset_path = os.path.join("TP2 and  3/services/epf-flower-data-science/src/datasets", dataset_filename)

    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file {dataset_filename} not found in the 'datasets' directory."
        )

    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse dataset {dataset_name}: {str(e)}"
        )
