import numpy as np
from typing import Optional, List, Union

class EmbeddingDB:
    def __init__(self, collection: str = 'embeddings.npz', embedding_dim: int = 768):
        self.collection: str = collection
        try:
            with np.load(self.collection, allow_pickle=True) as data:
                self.metadata: dict = data['metadata'].tolist()
                self.vectors: np.ndarray = data['vectors']
            print(f"Collection {collection} loaded!")
        except FileNotFoundError:
            self.metadata: dict = {}
            self.vectors: np.ndarray = np.empty((0, embedding_dim))
    
    def add_vector(self, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        self.vectors: np.ndarray = np.vstack((self.vectors, vector))
        idx: int = self.vectors.shape[0] - 1
        self.metadata[idx]: Optional[dict] = metadata

    def get_similar_vectors(self, vector: np.ndarray, top_k: int = 3) -> List[Optional[dict]]:
        sims: np.ndarray = self.compute_distance(vector, self.vectors)[0]
        top_k_idx: np.ndarray = np.argsort(sims)[::-1][:top_k]
        return [self.metadata[idx] for idx in top_k_idx]

    def compute_distance(self, vector: np.ndarray, array: np.ndarray) -> np.ndarray:
        sims: np.ndarray = vector @ array.T
        sims /= np.linalg.norm(vector) * np.linalg.norm(array, axis=1)
        return sims

    def save(self) -> None:
        with open(self.collection, 'wb') as f:
            np.savez(f, metadata=self.metadata, vectors=self.vectors)