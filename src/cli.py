"""Command-line interface exposed through Python Fire."""


class RagCLI:
    """RAG commands: index, search, answer, evaluate."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Chunk the corpus and persist the index."""
        print(f"[TODO] index {max_chunk_size=} {raw_dir=}")

    def search(self, query: str, k: int = 10) -> None:
        """Print the top-k sources for a single query."""
        print(f"[TODO] search {query=} {k=}")

    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int = 10
    ) -> None:
        """Search a whole dataset and save StudentSearchResults."""
        print(f"[TODO] search_dataset {dataset_path=} {k=}")

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single query from retrieved context."""
        print(f"[TODO] answer {query=} {k=}")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers and save StudentSearchResultsAndAnswer."""
        print(f"[TODO] answer_dataset {student_search_results_path=}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Report recall@k against a ground-truth dataset."""
        print(f"[TODO] evaluate {dataset_path=} {k=}")
