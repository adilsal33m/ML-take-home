from pydantic import BaseModel, Field
from typing import List


class SimilarityResponseSchema(BaseModel):
    url: List[str]
