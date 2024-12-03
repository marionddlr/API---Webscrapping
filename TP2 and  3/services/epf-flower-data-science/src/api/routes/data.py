import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from services.data import get_dataset, add_dataset, modify_dataset, load_dataset

router = APIRouter()

@router.get("/fetch-dataset/{dataset_name}")
async def fetch_dataset(dataset_name: str):
    try:
        dataset = get_dataset(dataset_name)
        return JSONResponse(content=dataset)
    except HTTPException as e:
        raise e
    

@router.post("/add-dataset/")
async def add_new_dataset(dataset_name: str, dataset_url: str):
    try:
        added_dataset = add_dataset(dataset_name, dataset_url)
        return {"message": f"Dataset {dataset_name} added successfully.", "dataset": added_dataset}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

 
@router.put("/update-dataset/{dataset_name}")
async def update_dataset(dataset_name: str, dataset_newurl: str):
    try:
        updated_dataset = modify_dataset(dataset_name, dataset_newurl)
        return {"message": f"Dataset {dataset_name} updated successfully.", "dataset": updated_dataset}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    
@router.get("/load-dataset/{dataset_name}")
async def fetch_dataset_to_json(dataset_name: str):
    try:
        df = load_dataset(dataset_name)
        dataset_content = df.to_dict(orient="records")
        return {
            "message": f"Dataset {dataset_name} loaded successfully from the URL.",
            "data": dataset_content
        }
    except HTTPException as e:
        raise e
