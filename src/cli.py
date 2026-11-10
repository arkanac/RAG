from src.chunking import Chunker
from src.indexing import Indexer
from src.retrieval import Retriever
from src.evaluation import evaluate
from .generation import TransformersEngine, answer_question, generate_answers


class RagCLI:
    """RAG commands: index, search, answer, evaluate."""

    def index(self, max_chunk_size: int = 1900, raw_dir: str = "data/raw",
              processed_dir: str = "data/processed") -> None:
        """Chunk the corpus and persist the index."""

        chunker = Chunker(max_chunk_size)
        chunks = chunker.chunk_corpus(raw_dir)
        indexer = Indexer()
        indexer.build_index(chunks, processed_dir)
        print(f"Ingestion complete! Indices saved under {processed_dir}")

    def search(self, query: str, k: int = 10,
               processed_dir: str = "data/processed") -> None:
        """Print the top-k sources for a single query."""
        retriever = Retriever(processed_dir)
        result = retriever.search(query, k)
        for source in result:
            print(f"Path: {source.file_path} | First/last index:"
                  f" {source.first_character_index}/"
                  f"{source.last_character_index}")

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10, processed_dir: str = "data/processed"
                       ) -> None:
        """Search a whole dataset and save StudentSearchResults."""
        retriever = Retriever(processed_dir)
        retriever.search_dataset(dataset_path, save_directory, k)

    def answer(self, query: str, k: int = 5,
               processed_dir: str = "data/processed") -> None:
        """Answer a single query using the retrieved context."""
        retriever = Retriever(processed_dir)
        sources = retriever.search(query, k)
        engine = TransformersEngine()
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
