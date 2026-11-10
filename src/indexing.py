from pathlib import Path
from src.models import Chunk
import bm25s
import json


class Indexer:
    def build_index(self, chunks: list[Chunk], processed_dir: str) -> None:
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        texts: list[str] = []
        texts += [(chunk.file_path + " " + chunk.text) for chunk in chunks]
        tokens = bm25s.tokenize(texts, stopwords="en")
        bm25 = bm25s.BM25()
        bm25.index(tokens)
        path = Path(processed_dir) / "bm25"
        bm25.save(path)
        c_path = Path(processed_dir) / "chunks.json"
        chunk_list: list[dict] = []
        with open(c_path, "w", encoding="utf-8") as f:
            chunk_list += [chunk.model_dump() for chunk in chunks]
            json.dump(chunk_list, f)
