from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from pathlib import Path
from typing import List,Literal,Dict
import asyncio
import uvicorn

app = FastAPI()

# Define the target directory
target_directory = Path(r'R:\\preds')

# Ensure the target directory exists
target_directory.mkdir(parents=True, exist_ok=True)

class Item(BaseModel):
    filename: str
    towrite_dicts: Dict

@app.post('/save')
async def save_json(item: Item):
    # Get the JSON data from the request body
    data = item.dict()

    # Define the target file path
    file_path = data['filename']
    file_path = target_directory / file_path
    data = data['towrite_dicts']
    
    try:
        # Write the JSON data to the file
        with file_path.open('w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        return {"message": f"Data saved to {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def main():
    pass

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")