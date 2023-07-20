import numpy as np
import torch

from typing import Any, Union
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor,AutoModel
from app.models.schemas.similarity import SimilarityResponseSchema
from app.utils.embedding_db import EmbeddingDB

imageType = Union[Image.Image, Any]

MODEL_BASE = "adilsaleem/vit-potatoes-plant-health-status"
MODEL_NAME = "./mlmodels/vit-potatoes-plant-health-status/"
DB_NAME = "./db/embeddings.npz"


class SimilarImagesFinder:
    def __init__(self):
        self.model_name = MODEL_NAME if Path(MODEL_NAME).exists() else MODEL_BASE
        self.model = AutoModel.from_pretrained(self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.db = EmbeddingDB(collection=DB_NAME)

        if not Path(MODEL_NAME).exists():
            self.classifier.save_pretrained(MODEL_NAME)

    def search(self, image: imageType) -> SimilarityResponseSchema:
        img_data = np.array(self.processor(image)['pixel_values'])
        with torch.no_grad():
            emb = self.model(torch.tensor(img_data)).last_hidden_state[:,0].numpy()

        result = self.db.get_similar_vectors(emb)
        return result
