import io
import time
from typing import Union, List

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.models.response import ResponseModel
from app.services.classifier import ImageClassifier

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
Classifier = ImageClassifier()


@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": f"Welcome to the image classifier server!!, servertime {time.time()}"
    }


@app.post("/classify", tags=["Image Classification"])
async def classify(file: List[Union[UploadFile, None]] = None):
    results = []
    for f in file:
        if not f:
            return ResponseModel(message="No file sent", success=False)

        content = await f.read()
        image = Image.open(io.BytesIO(content))
        results.append(Classifier.predict(image))
    
    return ResponseModel(data=results, message="Successful classification")
