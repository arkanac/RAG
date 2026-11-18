"""Semantic index built with FastEmbed."""
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticIndex:
    """Dense vector index over the chunks, computed on CPU."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        """Load the embedding model."""
        self.model_name = model_name
        self.model: Any = TextEmbedding(model_name=model_name)
        self.embeddings: np.ndarray | None = None

    def build(self, texts: list[str], save_dir: str) -> None:
        """Encode every chunk text and save the matrix with its metadata."""
        if not texts:
            raise ValueError("no text to embed")
        vectors = list(self.model.passage_embed(texts))
        matrix: np.ndarray = np.vstack(vectors).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "embeddings.npy", matrix)
        meta = {"model": self.model_name, "chunks": len(texts)}
        with open(folder / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self.embeddings = matrix

    def load(self, save_dir: str, expected_chunks: int) -> None:
        """Load a matrix saved by build and check it matches the chunks."""
        folder = Path(save_dir)
        matrix_path = folder / "embeddings.npy"
        meta_path = folder / "meta.json"
        if not matrix_path.is_file() or not meta_path.is_file():
            raise ValueError(f"no semantic index in {save_dir}, "
                             "run 'index --semantic' first")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model") != self.model_name:
            raise ValueError(f"semantic index built with {meta.get('model')}"
                             f", expected {self.model_name}")
        if meta.get("chunks") != expected_chunks:
            raise ValueError(f"semantic index has {meta.get('chunks')} "
                             f"chunks but the BM25 index has "
                             f"{expected_chunks}, run 'index --semantic' "
                             "again")
        self.embeddings = np.load(matrix_path)

    def search(self, query: str, k: int) -> list[int]:
        """Return the ids of the k chunks closest to the query."""
        if self.embeddings is None:
            raise ValueError("semantic index not loaded")
        vector = next(iter(self.model.query_embed(query)))
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.embeddings @ vector
        k = min(k, len(scores))
        return [int(i) for i in np.argsort(-scores)[:k]]

    @staticmethod
    def read_model(save_dir: str) -> str:
        """Return the model name stored with a semantic index."""
        meta_path = Path(save_dir) / "meta.json"
        if not meta_path.is_file():
            raise ValueError(f"no semantic index in {save_dir}, "
                             "run 'index --semantic' first")
        with open(meta_path, encoding="utf-8") as f:
            return str(json.load(f).get("model", ""))

    @staticmethod
    def remove(save_dir: str) -> None:
        """Delete a semantic index that no longer matches the chunks."""
        shutil.rmtree(save_dir, ignore_errors=True)
