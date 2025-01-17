from typing import Any, List, Union
from pathlib import Path
from PIL import Image
from transformers import pipeline,AutoImageProcessor,AutoModelForImageClassification

from app.models.schemas.classifier import ClassifierResponseSchema

imageType = Union[Image.Image, Any]

MODEL_BASE = "adilsaleem/vit-potatoes-plant-health-status"
MODEL_NAME = "./mlmodels/vit-potatoes-plant-health-status/"


class ImageClassifier:
    def __init__(self):
        self.model_name = MODEL_NAME if Path(MODEL_NAME).exists() else MODEL_BASE
        self.classifier = pipeline("image-classification", 
                                   model=AutoModelForImageClassification.from_pretrained(self.model_name),
                                   feature_extractor=AutoImageProcessor.from_pretrained(self.model_name))

        if not Path(MODEL_NAME).exists():
            self.classifier.save_pretrained(MODEL_NAME)

    def predict(self, image: imageType) -> List[ClassifierResponseSchema]:
        result = self.classifier(image)
        return result
