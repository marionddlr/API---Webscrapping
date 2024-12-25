import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from services.data import get_dataset, add_dataset, modify_dataset, load_dataset, process_dataset, split_dataset, train_model, predict_species, parameters_firestore, add_parameters_to_firestore, update_parameters_in_firestore

router = APIRouter()

@router.get("/fetch-dataset/{dataset_name}")
async def fetch_dataset(dataset_name: str):

    """
    Endpoint to fetch a dataset by its name.

    Args:
        dataset_name (str): The name of the dataset to fetch.

    Returns:
        JSONResponse: A JSON response containing the dataset if found.
    
    Raises:
        HTTPException: If the dataset is not found or there is an error in retrieving the dataset.
    """

    try:
        dataset = get_dataset(dataset_name)
        return JSONResponse(content=dataset)
    except HTTPException as e:
        raise e
    

@router.post("/add-dataset/")
async def add_new_dataset(dataset_name: str, dataset_url: str):

    """
    Endpoint to add a new dataset by providing its name and URL.

    Args:
        dataset_name (str): The name of the dataset to add.
        dataset_url (str): The URL from which to download the dataset.

    Returns:
        dict: A message indicating the success of the operation, and the dataset information.
    
    Raises:
        HTTPException: If there is an error in adding the dataset, such as invalid data.
    """

    try:
        added_dataset = add_dataset(dataset_name, dataset_url)
        return {"message": f"Dataset {dataset_name} added successfully.", "dataset": added_dataset}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

 
@router.put("/update-dataset/{dataset_name}")
async def update_dataset(dataset_name: str, dataset_newurl: str):

    """
    Endpoint to update the URL of an existing dataset.

    Args:
        dataset_name (str): The name of the dataset to update.
        dataset_newurl (str): The new URL for the dataset.

    Returns:
        dict: A message indicating the success of the operation, and the updated dataset information.
    
    Raises:
        HTTPException: If the dataset does not exist or if the URL update fails.
    """

    try:
        updated_dataset = modify_dataset(dataset_name, dataset_newurl)
        return {"message": f"Dataset {dataset_name} updated successfully.", "dataset": updated_dataset}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    
@router.get("/load-dataset/{dataset_name}")
async def fetch_dataset_to_json(dataset_name: str):

    """
    Endpoint to load a dataset and return its content as a JSON response.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        dict: A message indicating the success of the operation, and the dataset content as a list of records.
    
    Raises:
        HTTPException: If the dataset cannot be loaded or an error occurs during the loading process.
    """

    try:
        df = load_dataset(dataset_name)
        dataset_content = df.to_dict(orient="records")
        return {
            "message": f"Dataset {dataset_name} loaded successfully from the URL.",
            "data": dataset_content
        }
    except HTTPException as e:
        raise e


@router.get("/process-dataset/{dataset_name}")
async def process_dataset_endpoint(dataset_name: str):

    """
    Endpoint to process a dataset by removing the 'Iris-' prefix from the 'Species' column.

    Args:
        dataset_name (str): The name of the dataset to process.
    
    Returns:
        dict: A message indicating success, and the processed data.
    """

    try:
        df = load_dataset(dataset_name)
        df_processed = process_dataset(df)
        dataset_content = df_processed.to_dict(orient="records")
        return {
            "message": f"Dataset {dataset_name} processed successfully.",
            "data": dataset_content
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing dataset {dataset_name}: {str(e)}"
        )
    

@router.get("/split-dataset/{dataset_name}")
async def split_dataset_endpoint(dataset_name: str):
    """
    Endpoint pour diviser un dataset en ensembles d'entraînement et de test et retourner les deux sous forme JSON.

    Args:
        dataset_name (str): Le nom du dataset à diviser.
    
    Returns:
        dict: Un message de succès et les données divisées sous forme JSON.
    """
    try:
        df = load_dataset(dataset_name)
        df_processed = process_dataset(df)
        train_data, test_data = split_dataset(df_processed)
        train_data_content = train_data.to_dict(orient="records")
        test_data_content = test_data.to_dict(orient="records")
        return JSONResponse(
            content={
                "message": f"Dataset {dataset_name} divisé avec succès en ensembles d'entraînement et de test.",
                "train": train_data_content,
                "test": test_data_content
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue lors de la séparation du dataset {dataset_name}: {str(e)}"
        )


@router.post("/train-model/{dataset_name}")
async def train_model_endpoint(dataset_name: str):
    """
    Endpoint to train a classification model with a processed dataset.

    Args:
        dataset_name (str): The name of the dataset to use for training.

    Returns:
        dict: A success message and the path to the saved model.

    Raises:
        HTTPException: If any error occurs during the process.
    """
    try:
        df = load_dataset(dataset_name)
        processed_df = process_dataset(df)
        train_data, test_data = split_dataset(processed_df)
        X_train = train_data.drop(columns=["Species"])
        y_train = train_data["Species"]
        model_config_path = 'TP2 and  3/services/epf-flower-data-science/src/config/model_parameters.json'
        model_save_path = f'TP2 and  3/services/epf-flower-data-science/src/models/{dataset_name}_model.pkl'
        result = train_model(X_train, y_train, model_config_path, model_save_path)
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )
    

@router.post("/predict/{dataset_name}")
async def predict_with_model(dataset_name: str, input_data: dict):
    """
    Endpoint to make predictions with a trained classification model.

    Args:
        dataset_name (str): The name of the dataset used for training.
        input_data (dict): The input data for making predictions.

    Returns:
        dict: A dictionary containing the prediction result.
    
    Raises:
        HTTPException: If any error occurs during the prediction process.
    """
    try:
        result = predict_species(dataset_name, input_data)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during the prediction process: {str(e)}"
        )


@router.get("/get-parameters")
async def get_parameters():
    """Endpoint to retrieve parameters from Firestore.

    This endpoint calls the `parameters_firestore` function to fetch the parameters
    stored in Firestore and returns them in the response.

    Raises:
        HTTPException: If there is an error while retrieving the parameters from Firestore.

    Returns:
        dict: A dictionary containing the retrieved parameters.
    """
    try:
        params = parameters_firestore()
        return {"params": params}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des paramètres: {str(e)}")


# Endpoint to add parameters
@router.post("/add-parameters")
async def add_parameters(params: dict):
    """Endpoint to add parameters to Firestore.

    This endpoint accepts parameters in the form of a JSON object and stores
    them in Firestore under the 'parameters' document in the 'parameters' collection.

    Args:
        params (dict): A dictionary containing the new parameters.

    Raises:
        HTTPException: If an error occurs while adding the parameters to Firestore.

    Returns:
        dict: A success message and the added parameters.
    """
    try:
        result = add_parameters_to_firestore(params)
        return {"message": "Parameters added successfully", "data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding parameters: {str(e)}")


@router.put("/update-parameters")
async def update_parameters(params: dict):
    """Endpoint to update parameters in Firestore.

    This endpoint accepts parameters in the form of a JSON object and updates 
    the existing parameters in Firestore under the 'parameters' document.

    Args:
        params (dict): A dictionary containing the parameters to update.

    Raises:
        HTTPException: If an error occurs while updating the parameters in Firestore.

    Returns:
        dict: A success message and the updated parameters.
    """
    try:
        result = update_parameters_in_firestore(params)
        return {"message": "Parameters updated successfully", "data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating parameters: {str(e)}")
