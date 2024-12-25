import json
import sys
import os
import pandas as pd
import pytest
from fastapi import HTTPException
from google.cloud import firestore
from unittest.mock import patch
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.services.data import (
    get_dataset,
    add_dataset,
    modify_dataset,
    load_dataset,
    process_dataset,
    split_dataset,
    train_model,
    predict_species,
    parameters_firestore,
    add_parameters_to_firestore,
    update_parameters_in_firestore
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '/src/config/config.json')

@pytest.fixture
def mock_config_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_config_file:
        tmp_config_file.close()
        test_config = {
            "example_dataset": {
                "name": "example_dataset",
                "url": "http://example.com/dataset.csv"
            }
        }
        with open(tmp_config_file.name, 'w') as f:
            json.dump(test_config, f, indent=4)
        
        yield tmp_config_file.name
        os.remove(tmp_config_file.name)

def test_get_dataset(mock_config_file):
    with patch('CONFIG_PATH', mock_config_file):
        dataset = get_dataset("example_dataset")
        assert dataset == {
            "name": "example_dataset",
            "url": "http://example.com/dataset.csv"
        }

def test_get_dataset_not_found():
    with pytest.raises(HTTPException) as exc_info:
        get_dataset("non_existent_dataset")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Dataset non_existent_dataset not found."

def test_add_dataset(mock_config_file):
    with patch('src.services.data.CONFIG_PATH', mock_config_file):
        new_dataset = add_dataset("new_dataset", "http://newexample.com/dataset.csv")
        assert new_dataset == {
            "name": "new_dataset",
            "url": "http://newexample.com/dataset.csv"
        }

def test_add_existing_dataset(mock_config_file):
    with patch('src.services.data.CONFIG_PATH', mock_config_file):
        add_dataset("existing_dataset", "http://example.com/existing_dataset.csv")
        with pytest.raises(ValueError) as exc_info:
            add_dataset("existing_dataset", "http://example.com/existing_dataset.csv")
        assert str(exc_info.value) == "Dataset existing_dataset already exists in the config."

def test_modify_dataset(mock_config_file):
    with patch('src.services.data.CONFIG_PATH', mock_config_file):
        modify_dataset("existing_dataset", "http://example.com/modified_dataset.csv")
        with open(mock_config_file, 'r') as f:
            config = json.load(f)
        assert config["existing_dataset"]["url"] == "http://example.com/modified_dataset.csv"

def test_load_dataset(mock_config_file):
    with patch('src.services.data.CONFIG_PATH', mock_config_file):
        df = load_dataset("example_dataset")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

def test_load_dataset_not_found():
    with pytest.raises(HTTPException) as exc_info:
        load_dataset("non_existent_dataset")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Dataset file non_existent_dataset.csv not found in the 'datasets' directory."

def test_process_dataset():
    df = pd.DataFrame({
        "Id": [1, 2, 3],
        "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        "SepalLength": [5.1, 7.0, 6.3],
        "SepalWidth": [3.5, 3.2, 3.3],
        "PetalLength": [1.4, 4.7, 6.0],
        "PetalWidth": [0.2, 1.4, 2.5]
    })
    processed_df = process_dataset(df)
    assert 'Species' in processed_df.columns
    assert not processed_df['Species'].str.startswith('Iris-').any()
    assert 'Id' not in processed_df.columns

def test_split_dataset():
    df = pd.DataFrame({
        "Species": ["setosa", "versicolor", "virginica"],
        "sepal_length": [5.1, 7.0, 6.3],
        "sepal_width": [3.5, 3.2, 3.3],
        "petal_length": [1.4, 4.7, 6.0],
        "petal_width": [0.2, 1.4, 2.5]
    })
    train_data, test_data = split_dataset(df)
    assert len(train_data) + len(test_data) == len(df)
    assert 'Species' in train_data.columns and 'Species' in test_data.columns

def test_train_model():
    # Préparation des données de test
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6]
    })
    y_train = pd.Series(["class1", "class2", "class3"])
    model_config_path = 'TP2 and  3/services/epf-flower-data-science/src/config/model_parameters.json'
    save_path = 'model_test.pkl'
    result = train_model(X_train, y_train, model_config_path, save_path)
    assert result['message'] == 'Model trained and saved successfully.'
    assert os.path.exists(save_path)
    os.remove(save_path)

def test_predict_species():
    X_train = pd.DataFrame({
        "feature1": [1],
        "feature2": [2]
    })
    y_train = pd.Series(["test_class"])
    model_config_path = 'TP2 and  3/services/epf-flower-data-science/src/config/model_parameters.json'
    save_path = 'model_test_predict.pkl'
    train_model(X_train, y_train, model_config_path, save_path)

    result = predict_species("test_dataset", {"feature1": 1, "feature2": 2})
    assert result == {"predicted_class": "test_class"}
    os.remove(save_path)

def test_parameters_firestore():
    params = parameters_firestore()
    assert isinstance(params, dict)

def test_add_parameters_to_firestore():
    new_params = {'test_key': 'test_value'}
    updated_params = add_parameters_to_firestore(new_params)
    assert updated_params == new_params

def test_update_parameters_in_firestore():
    updated_params = {'test_key': 'updated_value'}
    params = update_parameters_in_firestore(updated_params)
    assert params == updated_params
