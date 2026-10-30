import bm25s
from pathlib import Path
import json
from models import Chunk, MinimalSource


class Retriever:
    def __init__(self, processed_dir: str) -> None:
        self.bm25 = bm25s.BM25.load(Path(processed_dir) / "bm25")
        self.chunks = []
        with open(Path(processed_dir) / "chunks.json", "r", encoding="utf-8"
                  ) as f:
            chunk_list: list[dict] = json.load(f)
            self.chunks += [Chunk.model_validate(chunk) for chunk
                            in chunk_list]
            
    def search(self, query: str, k: int):
        tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        test = self.bm25.retrieve(tokens, k=k)
        return test

def test():
    re = Retriever("data/processed")
    print(re.search("What are the default values for FP8_MIN and"
                    "FP8_MAX constants in vLLM's triton_flash_"
                    "attention module?", 10)[0])


if "__main__" == __name__:
    test()