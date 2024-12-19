from fastapi import HTTPException
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from google.cloud import firestore

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


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the dataset by removing the 'Iris-' prefix from the 'Species' column.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The processed dataset with updated 'Species' column.
    """
    if 'Species' in df.columns:
        df['Species'] = df['Species'].str.replace('Iris-', '', regex=False)
        df = df.drop(columns=["Id"])
    return df


def split_dataset(df: pd.DataFrame):
    """
    Function to split a dataset into training and test sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.

    Returns:
        tuple: Two DataFrames, one for the training set and one for the test set.
    """
    try:
        X = df.drop(columns=['Species'])
        y = df['Species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_data = X_train.copy()
        train_data['Species'] = y_train
        test_data = X_test.copy()
        test_data['Species'] = y_test
        return train_data, test_data
    except Exception as e:
        raise ValueError(f"Erreur lors de la séparation du dataset : {str(e)}")


def train_model(X_train, y_train, model_config_path, save_path):
    """
    Train a logistic regression model with the training data and save the model.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        model_config_path (str): The path to the JSON file containing the model parameters.
        save_path (str): The path to save the trained model.

    Raises:
        HTTPException: If an error occurs during training or saving.

    Returns:
        dict: Success message and the path to the saved model.
    """
    try:
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        model = LogisticRegression(**model_config['parameters'])
        model.fit(X_train, y_train)
        joblib.dump(model, save_path)
        return {
            "message": "Model trained and saved successfully.",
            "model_path": save_path
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during training or saving the model: {str(e)}"
        )
    

def predict_species(dataset_name: str, input_data: dict):
    """
    Make predictions using a trained classification model.

    Args:
        dataset_name (str): The name of the dataset used for training.
        input_data (dict): The input data for making predictions.

    Returns:
        dict: A dictionary with the predicted class and any necessary metadata.

    Raises:
        HTTPException: If the model is not found or there is an error in the prediction process.
    """
    try:
        model_path = f'TP2 and  3/services/epf-flower-data-science/src/models/{dataset_name}_model.pkl'
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        model = joblib.load(model_path)
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return {"predicted_class": prediction[0]}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during the prediction process: {str(e)}"
        )
    

def parameters_firestore():
    """Retrieve parameters from Firestore.

    This function connects to Firestore using a service account key, retrieves 
    the parameters document from the "parameters" collection, and returns its content.

    Raises:
        HTTPException: If there is an issue with connecting to Firestore or 
        retrieving the parameters.

    Returns:
        dict: A dictionary containing the parameters stored in Firestore.
    """
    credentials_path = 'TP2 and  3/services/epf-flower-data-science/src/config//google.json'
    try:
        db = firestore.Client.from_service_account_json(credentials_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la connexion à Firestore: {str(e)}")
    try:
        doc_ref = db.collection('parameters').document('parameters')
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Les paramètres n'ont pas été trouvés dans Firestore.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des paramètres : {str(e)}")
    

def add_parameters_to_firestore(new_params: dict):
    """Add new parameters to Firestore.

    This function will add a new document or update an existing document in 
    the Firestore collection 'parameters' with the provided parameters.

    Args:
        new_params (dict): The dictionary containing parameters to add or update.

    Raises:
        HTTPException: If there is an error connecting to Firestore or saving data.

    Returns:
        dict: The parameters that were added or updated in Firestore.
    """
    credentials_path = 'TP2 and  3/services/epf-flower-data-science/src/config/google.json'
    try:
        db = firestore.Client.from_service_account_json(credentials_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Firestore: {str(e)}")
    try:
        doc_ref = db.collection('parameters').document('parameters')
        doc_ref.set(new_params)
        return new_params
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding parameters to Firestore: {str(e)}")


def update_parameters_in_firestore(updated_params: dict):
    """Update parameters in Firestore.

    This function updates an existing document in Firestore under the 
    'parameters' collection.

    Args:
        updated_params (dict): The dictionary of parameters to update.

    Raises:
        HTTPException: If there is an error updating the parameters in Firestore.

    Returns:
        dict: The updated parameters in Firestore.
    """
    credentials_path = 'TP2 and  3/services/epf-flower-data-science/src/config/google.json'
    try:
        db = firestore.Client.from_service_account_json(credentials_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Firestore: {str(e)}")
    try:
        doc_ref = db.collection('parameters').document('parameters')
        doc_ref.update(updated_params)
        return updated_params
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating parameters in Firestore: {str(e)}")

