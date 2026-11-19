"""Command-line interface exposed through Python Fire."""
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .chunking import Chunker
from .embedding import EMBEDDING_MODEL
from .evaluation import evaluate as compute_recall
from .generation import (MAX_CONTEXT_SOURCES, TransformersEngine,
                         answer_question, generate_answers, read_source)
from .indexing import Indexer
from .models import MinimalSource
from .retrieval import Retriever
from .validation import validate_chunk_size, validate_k, validate_method

CONSOLE = Console()
PREVIEW_LENGTH = 50


def sources_table(sources: list[MinimalSource], title: str,
                  used: int = 0) -> Table:
    """Build a table of sources.

    Args:
        sources: The sources to display, best first.
        title: Title of the table.
        used: Number of leading sources given to the model; they are shown
            in bold and the others are dimmed. 0 shows every row normally.

    Returns:
        A rich table ready to be printed.
    """
    table = Table(title=title, title_justify="left")
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Characters", justify="right")
    table.add_column("Preview", overflow="ellipsis", no_wrap=True)
    for i, source in enumerate(sources, 1):
        lines = read_source(source).strip().splitlines() or [""]
        in_context = i <= used
        table.add_row(f"[bold]{i}[/bold]" if in_context else str(i),
                      source.file_path.removeprefix("data/raw/"),
                      f"{source.first_character_index}-"
                      f"{source.last_character_index}",
                      lines[0][:PREVIEW_LENGTH],
                      style="dim" if used and not in_context else None)
    return table


class RagCLI:
    """RAG commands: index, search, answer, evaluate and the HTTP API."""

    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw",
              processed_dir: str = "data/processed",
              semantic: bool = False,
              embedding_model: str = EMBEDDING_MODEL) -> None:
        """Chunk the corpus and persist the index.

        Args:
            max_chunk_size: Maximum chunk length in characters (1-2000).
            raw_dir: Directory of the corpus.
            processed_dir: Directory where the index is saved.
            semantic: Also build the semantic index (bonus).
            embedding_model: FastEmbed model used for the semantic index.
        """
        size = validate_chunk_size(max_chunk_size)
        start = time.perf_counter()
        chunks = Chunker(size).chunk_corpus(raw_dir)
        if not chunks:
            raise ValueError(f"no supported files found in {raw_dir}")
        Indexer().build_index(chunks, processed_dir, semantic=semantic,
                              embedding_model=embedding_model)
        elapsed = time.perf_counter() - start
        print(f"Ingestion complete! Indices saved under {processed_dir}")
        details = f"{len(chunks)} chunks | max {size} characters"
        if semantic:
            details += f" | semantic: {embedding_model}"
        CONSOLE.print(f"[dim]{details} | {elapsed:.1f} s[/dim]")

    def search(self, query: str, k: int = 10,
               processed_dir: str = "data/processed",
               method: str = "bm25") -> None:
        """Print the top-k sources for a single query.

        Args:
            query: The question.
            k: Number of sources to return.
            processed_dir: Directory of the index.
            method: Retrieval method: bm25, semantic or hybrid.
        """
        k = validate_k(k)
        method = validate_method(method)
        start = time.perf_counter()
        with CONSOLE.status("Searching the index..."):
            sources = Retriever(processed_dir).search(query, k, method)
        elapsed = time.perf_counter() - start
        CONSOLE.print(sources_table(sources, f"{query}  [dim]({method}, "
                                             f"{elapsed:.2f} s)[/dim]"))

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10, processed_dir: str = "data/processed",
                       method: str = "bm25") -> None:
        """Search a whole dataset and save StudentSearchResults.

        Args:
            dataset_path: JSON file of questions.
            save_directory: Directory where the results are written.
            k: Number of sources per question.
            processed_dir: Directory of the index.
            method: Retrieval method: bm25, semantic or hybrid.
        """
        k = validate_k(k)
        method = validate_method(method)
        start = time.perf_counter()
        retriever = Retriever(processed_dir)
        retriever.search_dataset(dataset_path, save_directory, k, method)
        elapsed = time.perf_counter() - start
        CONSOLE.print(f"[dim]{method} | k={k} | {elapsed:.1f} s[/dim]")

    def answer(self, query: str, k: int = 5,
               processed_dir: str = "data/processed",
               method: str = "bm25") -> None:
        """Answer a single query using the retrieved context.

        Args:
            query: The question.
            k: Number of sources to retrieve.
            processed_dir: Directory of the index.
            method: Retrieval method: bm25, semantic or hybrid.
        """
        k = validate_k(k)
        method = validate_method(method)
        start = time.perf_counter()
        with CONSOLE.status("Searching the index..."):
            sources = Retriever(processed_dir).search(query, k, method)
        with CONSOLE.status("Loading Qwen and generating the answer..."):
            engine = TransformersEngine()
            text = answer_question(engine, query, sources)
        elapsed = time.perf_counter() - start
        CONSOLE.print(Panel(Markdown(text), title=f"[bold]{query}[/bold]",
                            subtitle=f"{method} | k={k} | {engine.device}"
                                     f" | {elapsed:.1f} s",
                            border_style="green"))
        CONSOLE.print(sources_table(sources, "Sources",
                                    used=MAX_CONTEXT_SOURCES))
        CONSOLE.print(f"[dim]Rows in bold: the passages given to the "
                      f"model (first {MAX_CONTEXT_SOURCES}).[/dim]")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Generate answers for a whole search results file.

        Args:
            student_search_results_path: Output file of search_dataset.
            save_directory: Directory where the answers are written.
        """
        start = time.perf_counter()
        engine = TransformersEngine()
        CONSOLE.print(f"[dim]Device: {engine.device}[/dim]")
        generate_answers(engine, student_search_results_path, save_directory)
        elapsed = time.perf_counter() - start
        CONSOLE.print(f"[dim]{elapsed:.1f} s[/dim]")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Report recall@k of a search results file.

        Args:
            student_search_results_path: Output file of search_dataset.
            dataset_path: Matching AnsweredQuestions dataset.
        """
        results = compute_recall(student_search_results_path, dataset_path)
        table = Table(title="Recall@k", title_justify="left")
        table.add_column("k", justify="right")
        table.add_column("Recall", justify="right")
        for k, value in results.items():
            table.add_row(str(k), f"{value:.3f} ({value:.1%})")
        CONSOLE.print(table)

    def api(self, host: str = "127.0.0.1", port: int = 8000,
            processed_dir: str = "data/processed") -> None:
        """Start the local HTTP API (bonus).

        Args:
            host: Address to listen on; 127.0.0.1 keeps it local.
            port: Port to listen on.
            processed_dir: Directory of the index.
        """
        import uvicorn

        from .api import create_app

        app = create_app(processed_dir)
        CONSOLE.print(f"[green]API on http://{host}:{port}[/green]  "
                      f"[dim]interactive docs: /docs[/dim]")
        uvicorn.run(app, host=host, port=port)
