from typing import Union

from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put('/get_characters/')
async def get_prediction(image: UploadFile):
    return [
        {'name': 'luffy', 'conf': 0.5},
        {'name': 'nami', 'conf': 0.1},
        {'name': 'sanji', 'conf': 0.4},
        {'name': image.filename, 'conf': 0},
    ]
