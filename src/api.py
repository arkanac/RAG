"""Local HTTP API exposing search and answer generation."""
import uuid

from fastapi import FastAPI, HTTPException

from .generation import TransformersEngine, answer_question
from .models import MinimalAnswer, MinimalSearchResults, HealthStatus
from .retrieval import Retriever
from .validation import validate_k


def create_app(processed_dir: str) -> FastAPI:
    """Build the FastAPI application.

    The index is loaded once at startup. The Qwen model is loaded on the
    first call to /answer only, so the API starts fast and /search never
    waits for it.

    Args:
        processed_dir: Directory holding the index built by `index`.

    Returns:
        The FastAPI application, ready to be served by uvicorn.
    """
    app = FastAPI(title="RAG against the machine")
    retriever = Retriever(processed_dir)
    engines: dict[str, TransformersEngine] = {}

    @app.get("/health")
    def health() -> HealthStatus:
        """Report that the server runs, with the index size and model state."""
        return HealthStatus(status="ok",
                            chunks=len(retriever.chunks),
                            model_loaded="cpu" in engines)

    @app.get("/search")
    def search(query: str, k: int = 10) -> MinimalSearchResults:
        """Return the top-k sources for a question.

        Args:
            query: The question.
            k: Number of sources to return (at least 1).

        Returns:
            The question, a generated id and its retrieved sources.

        Raises:
            HTTPException: 400 if k is invalid.
        """
        try:
            validate_k(k)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        sources = retriever.search(query, k)
        return MinimalSearchResults(question_id=str(uuid.uuid4()),
                                    question=query,
                                    retrieved_sources=sources)

    @app.get("/answer")
    def answer(query: str, k: int = 5) -> MinimalAnswer:
        """Answer a question from its retrieved sources.

        The model runs on CPU and is loaded on the first call only.

        Args:
            query: The question.
            k: Number of sources to retrieve (at least 1).

        Returns:
            The question, a generated id, its sources and the answer.

        Raises:
            HTTPException: 400 if k is invalid.
        """
        try:
            validate_k(k)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        if "cpu" not in engines:
            engines["cpu"] = TransformersEngine(device="cpu")
        engine = engines["cpu"]
        sources = retriever.search(query, k)
        text = answer_question(engine, query, sources)
        return MinimalAnswer(question_id=str(uuid.uuid4()),
                             question=query,
                             retrieved_sources=sources,
                             answer=text)

    return app
