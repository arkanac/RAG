import bm25s
from pathlib import Path
import json
from tqdm import tqdm
from src.models import (Chunk, MinimalSource, RagDataset, MinimalSearchResults,
                        StudentSearchResults)


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

    def search_dataset(self, dataset_path: str, save_directory: str, k: int
                       ) -> None:
        with open(dataset_path, encoding="utf-8") as f:
            rag_questions = json.load(f)
            dataset = RagDataset.model_validate(rag_questions)
            msr_list: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions):
            answer = self.search(question.question, k)
            min_srch_r = MinimalSearchResults(question_id=question.question_id,
                                              question=question.question,
                                              retrieved_sources=answer)
            msr_list.append(min_srch_r)
        stud_search_r = StudentSearchResults(search_results=msr_list, k=k)
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        path1 = Path(save_directory)
        path2 = Path(dataset_path).name
        with open(path1 / path2, "w",
                  encoding="utf-8") as f:
            f.write(stud_search_r.model_dump_json(indent=2))
        print(f"Saved student_research_results to {path1}/{path2}")
