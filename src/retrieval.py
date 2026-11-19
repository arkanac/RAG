import bm25s
from pathlib import Path
import json
from tqdm import tqdm
from src.models import (Chunk, MinimalSource, RagDataset, MinimalSearchResults,
                        StudentSearchResults)
from .embedding import SemanticIndex

RRF_K = 60
RRF_CANDIDATES = 50


class Retriever:
    def __init__(self, processed_dir: str) -> None:
        if not (Path(processed_dir) / "bm25").is_dir():
            raise ValueError(f"no index found in {processed_dir}, "
                             "run the 'index' command first")
        self.bm25 = bm25s.BM25.load(Path(processed_dir) / "bm25")
        self.chunks = []
        with open(Path(processed_dir) / "chunks.json", "r", encoding="utf-8"
                  ) as f:
            chunk_list: list[dict] = json.load(f)
            self.chunks += [Chunk.model_validate(chunk) for chunk
                            in chunk_list]
        self.processed_dir = processed_dir
        self.semantic: SemanticIndex | None = None

    def _semantic_index(self) -> SemanticIndex:
        """Load the semantic index on first use, with its own model."""
        if self.semantic is None:
            folder = str(Path(self.processed_dir) / "semantic")
            index = SemanticIndex(SemanticIndex.read_model(folder))
            index.load(folder, len(self.chunks))
            self.semantic = index
        return self.semantic

    def _bm25_ids(self, query: str, k: int) -> list[int]:
        """Return the ids of the k best chunks according to BM25."""
        tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        results, _ = self.bm25.retrieve(tokens, k=k)
        return [int(i) for i in results[0]]

    def search(self, query: str, k: int,
               method: str = "bm25") -> list[MinimalSource]:
        """Return the top-k sources for a query."""
        k = min(k, len(self.chunks))
        if method == "semantic":
            ids = self._semantic_index().search(query, k)
        elif method == "hybrid":
            ids = self._hybrid_ids(query, k)
        else:
            ids = self._bm25_ids(query, k)
        return [MinimalSource(file_path=self.chunks[i].file_path,
                              first_character_index=self.chunks[i]
                              .first_character_index,
                              last_character_index=self.chunks[i]
                              .last_character_index)
                for i in ids]

    def search_dataset(self, dataset_path: str, save_directory: str, k: int,
                       method: str = "bm25") -> None:
        """Search every question of a dataset and save the results."""
        with open(dataset_path, encoding="utf-8") as f:
            dataset = RagDataset.model_validate(json.load(f))
        results = [MinimalSearchResults(
                       question_id=q.question_id, question=q.question,
                       retrieved_sources=self.search(q.question, k, method))
                   for q in tqdm(dataset.rag_questions)]
        output = StudentSearchResults(search_results=results, k=k)
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(dataset_path).name
        out_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results to {out_path}")

    def _hybrid_ids(self, query: str, k: int) -> list[int]:
        """Fuse the BM25 and semantic rankings with Reciprocal Rank Fusion."""
        n = min(max(k, RRF_CANDIDATES), len(self.chunks))
        rankings = (self._bm25_ids(query, n),
                    self._semantic_index().search(query, n))
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, 1):
                points = 1 / (RRF_K + rank)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + points
        return sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
