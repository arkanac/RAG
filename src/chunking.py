from pathlib import Path
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from src.models import Chunk
from tqdm import tqdm

CODE_DIRS = {"vllm"}
DOC_DIRS = {"docs"}


class Chunker:
    """Split the corpus files into chunks with exact character offsets."""

    def __init__(self, max_chunk_size: int) -> None:
        """Create one splitter per supported file type."""
        self.max_chunk_size = max_chunk_size
        self.splitters = self._make_splitters(max_chunk_size)

    @staticmethod
    def _is_useful(path: Path, root: Path) -> bool:
        """Keep only the files that can contain an answer."""
        parts = path.relative_to(root).parts
        top_dir = parts[1] if len(parts) > 2 else ""
        if path.suffix == ".py":
            return top_dir in CODE_DIRS
        return len(parts) == 2 or top_dir in DOC_DIRS

    def find_files(self, raw_dir: str) -> dict[str, list[Path]]:
        """Return the useful files of raw_dir, sorted, grouped by extension."""
        data = Path(raw_dir)
        if not data.is_dir():
            raise FileNotFoundError(f"{raw_dir} is not a directory")
        files: dict[str, list[Path]] = {".py": [], ".md": [], ".txt": []}
        for file in sorted(data.rglob("*")):
            if (file.suffix in files and file.is_file()
                    and self._is_useful(file, data)):
                files[file.suffix].append(file)
        return files

    @staticmethod
    def _make_splitters(max_chunk_size: int
                        ) -> dict[str, RecursiveCharacterTextSplitter]:
        """Build the Python and Markdown splitters with a 10 % overlap."""
        def make(language: Language) -> RecursiveCharacterTextSplitter:
            return RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=max_chunk_size,
                chunk_overlap=max_chunk_size // 10, add_start_index=True)
        markdown = make(Language.MARKDOWN)
        return {".py": make(Language.PYTHON), ".md": markdown,
                ".txt": markdown}

    def chunk_file(self, path: Path) -> list[Chunk]:
        """Split one file into chunks."""
        text = path.read_text(encoding="utf-8", errors="replace")
        docs = self.splitters[path.suffix].create_documents([text])
        return [Chunk(file_path=str(path),
                      first_character_index=doc.metadata["start_index"],
                      last_character_index=(doc.metadata["start_index"]
                                            + len(doc.page_content)),
                      text=doc.page_content)
                for doc in docs if doc.metadata["start_index"] != -1]

    def chunk_corpus(self, raw_dir: str) -> list[Chunk]:
        """Chunk every useful file of the corpus."""
        chunks: list[Chunk] = []
        for extension, paths in self.find_files(raw_dir).items():
            for path in tqdm(paths, desc=extension):
                try:
                    chunks.extend(self.chunk_file(path))
                except OSError as err:
                    print(f"{err} at path {path}")
        return chunks
