import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from services.data import get_dataset, add_dataset, modify_dataset, load_dataset, process_dataset, split_dataset

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
