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

    def search(self, query: str, k: int) -> list[MinimalSource]:
        tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        ids, _ = self.bm25.retrieve(tokens, k=k)
        source_list: list[MinimalSource] = []
        for num in ids[0]:
            chunk = self.chunks[num]
            min_s = MinimalSource(file_path=chunk.file_path,
                                  first_character_index=chunk.
                                  first_character_index,
                                  last_character_index=chunk.
                                  last_character_index)
            source_list.append(min_s)
        return source_list


def test():
    re = Retriever("data/processed")
    print(re.search("What are the default values for FP8_MIN and"
                    "FP8_MAX constants in vLLM's triton_flash_"
                    "attention module?", 10)[0])
if "__main__" == __name__:
    test()