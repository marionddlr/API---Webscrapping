import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', '..')))
from src.app import get_application

@pytest.fixture
def client():
    app = get_application()
    return TestClient(app)

def test_fetch_dataset(client):
    dataset_name = "example_dataset"
    response = client.get(f"/fetch-dataset/{dataset_name}")
    assert response.status_code == 200 or response.status_code == 404
    if response.status_code == 200:
        assert "data" in response.json()

def test_add_new_dataset(client):
    dataset_name = "new_dataset"
    dataset_url = "https://example.com/dataset.csv"
    response = client.post(f"/add-dataset/?dataset_name={dataset_name}&dataset_url={dataset_url}")
    assert response.status_code == 200 or response.status_code == 400

def test_update_dataset(client):
    dataset_name = "existing_dataset"
    dataset_newurl = "https://example.com/new_dataset.csv"
    response = client.put(f"/update-dataset/?dataset_name={dataset_name}", json={"dataset_newurl": dataset_newurl})
    assert response.status_code == 200 or response.status_code == 404
    if response.status_code == 200:
        json_response = response.json()
        assert "message" in json_response
        assert "dataset" in json_response

def test_load_dataset(client):
    dataset_name = "example_dataset"
    response = client.get(f"/load-dataset/{dataset_name}")
    assert response.status_code == 200 or response.status_code == 404
    if response.status_code == 200:
        json_response = response.json()
        assert "data" in json_response

def test_process_dataset(client):
    dataset_name = "example_dataset"
    response = client.get(f"/process-dataset/{dataset_name}")
    assert response.status_code == 200 or response.status_code == 404
    if response.status_code == 200:
        json_response = response.json()
        assert "data" in json_response

def test_split_dataset(client):
    dataset_name = "example_dataset"
    response = client.get(f"/split-dataset/{dataset_name}")
    assert response.status_code == 200 or response.status_code == 404
    if response.status_code == 200:
        json_response = response.json()
        assert "train" in json_response
        assert "test" in json_response

def test_train_model(client):
    dataset_name = "example_dataset"
    response = client.post(f"/train-model/{dataset_name}")
    assert response.status_code in [200, 404]  
    assert "message" in response.json()

def test_predict_with_model(client):
    dataset_name = "example_dataset"
    input_data = {"feature1": 1.0, "feature2": 2.0}
    response = client.post(f"/predict/{dataset_name}", json=input_data)
    assert response.status_code == 200 or response.status_code == 500
    if response.status_code == 200:
        json_response = response.json()
        assert "prediction" in json_response

def test_get_parameters(client):
    response = client.get("/get-parameters")
    assert response.status_code == 200 or response.status_code == 500
    if response.status_code == 200:
        json_response = response.json()
        assert "params" in json_response

def test_add_parameters(client):
    params = {"param1": 10, "param2": "value"}
    response = client.post("/add-parameters", json=params)
    assert response.status_code == 200 or response.status_code == 500
    if response.status_code == 200:
        json_response = response.json()
        assert "message" in json_response

def test_update_parameters(client):
    params = {"param1": 20, "param2": "new_value"}
    response = client.put("/update-parameters", json=params)
    assert response.status_code == 200 or response.status_code == 500
    if response.status_code == 200:
        json_response = response.json()
        assert "message" in json_response
