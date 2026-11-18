import uvicorn
from src.chunking import Chunker
from src.indexing import Indexer
from src.retrieval import Retriever
from src.evaluation import evaluate
from .generation import TransformersEngine, answer_question, generate_answers
from .embedding import EMBEDDING_MODEL
from .validation import validate_chunk_size, validate_k, validate_method
from .api import create_app


class RagCLI:
    """RAG commands: index, search, answer, evaluate."""

    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw",
              processed_dir: str = "data/processed",
              semantic: bool = False,
              embedding_model: str = EMBEDDING_MODEL) -> None:
        """Chunk the corpus and persist the index."""
        size = validate_chunk_size(max_chunk_size)
        chunks = Chunker(size).chunk_corpus(raw_dir)
        if not chunks:
            raise ValueError(f"no supported files found in {raw_dir}")
        Indexer().build_index(chunks, processed_dir, semantic=semantic,
                              embedding_model=embedding_model)
        print(f"Ingestion complete! Indices saved under {processed_dir}")

    def search(self, query: str, k: int = 10,
               processed_dir: str = "data/processed",
               method: str = "bm25") -> None:
        """Print the top-k sources for a single query."""
        k = validate_k(k)
        method = validate_method(method)
        retriever = Retriever(processed_dir)
        for source in retriever.search(query, k, method):
            print(f"Path: {source.file_path} | First/last index: "
                  f"{source.first_character_index}/"
                  f"{source.last_character_index}")

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10, processed_dir: str = "data/processed",
                       method: str = "bm25") -> None:
        """Search a whole dataset and save StudentSearchResults."""
        k = validate_k(k)
        method = validate_method(method)
        retriever = Retriever(processed_dir)
        retriever.search_dataset(dataset_path, save_directory, k, method)

    def answer(self, query: str, k: int = 5,
               processed_dir: str = "data/processed",
               method: str = "bm25") -> None:
        """Answer a single query using the retrieved context."""
        k = validate_k(k)
        method = validate_method(method)
        retriever = Retriever(processed_dir)
        sources = retriever.search(query, k, method)
        engine = TransformersEngine()
        print(f"Using device: {engine.device}")
        print(answer_question(engine, query, sources))
        print("\nSources:")
        for source in sources:
            print(f"- {source.file_path} [{source.first_character_index}:"
                  f"{source.last_character_index}]")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers for a whole search results file."""
        engine = TransformersEngine()
        generate_answers(engine, student_search_results_path, save_directory)

    def evaluate(self, student_search_results_path: str, dataset_path: str,
                 k: int = 10) -> None:
        """Report recall@k against a ground-truth dataset."""
        evaluate(student_search_results_path, dataset_path)

    def api(self, host: str = "127.0.0.1", port: int = 8000,
            processed_dir: str = "data/processed") -> None:
        app = create_app(processed_dir)
        uvicorn.run(app, host=host, port=port)
