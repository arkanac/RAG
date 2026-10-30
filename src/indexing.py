from pathlib import Path
from src.models import Chunk
from src.chunking import Chunker
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


def test():
    ch = Chunker(2000)
    chunks = ch.chunk_corpus("data/raw")
    ind = Indexer()
    ind.build_index(chunks, "data/processed")
    question = bm25s.tokenize("What are the default values for FP8_MIN and"
                              "FP8_MAX constants in vLLM's triton_flash_"
                              "attention module?", stopwords="en",
                              show_progress=False)
    ids, score = (ind.retriever.retrieve(question, k=10))


if "__main__" == __name__:
    test()
