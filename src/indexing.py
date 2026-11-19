from pathlib import Path
from src.models import Chunk
from .embedding import EMBEDDING_MODEL, SemanticIndex
import bm25s
import json


class Indexer:
    """Build and save the BM25 index, and optionally the semantic one."""

    def build_index(self, chunks: list[Chunk], processed_dir: str,
                    semantic: bool = False,
                    embedding_model: str = EMBEDDING_MODEL) -> None:
        """Index the chunks and save everything under processed_dir."""
        out_dir = Path(processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        texts = [f"{chunk.file_path} {chunk.text}" for chunk in chunks]
        bm25 = bm25s.BM25()
        bm25.index(bm25s.tokenize(texts, stopwords="en"))
        bm25.save(out_dir / "bm25")
        with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([chunk.model_dump() for chunk in chunks], f)
        semantic_dir = str(out_dir / "semantic")
        if semantic:
            SemanticIndex(embedding_model).build(texts, semantic_dir)
        else:
            SemanticIndex.remove(semantic_dir)
