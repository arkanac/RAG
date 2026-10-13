from pathlib import Path
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from src.models import Chunk


def find_files(raw_dir: str) -> dict[str, list[Path]]:
    data = Path(raw_dir)
    if not data.is_dir():
        print("No data specified")
        return {}
    data_dict: dict[str, list[Path]] = {".py": [], ".md": [], ".txt": []}
    for file in data.rglob("*"):
        if file.is_file():
            s = file.suffix
            if s in data_dict:
                data_dict[s].append(file)
    for key in data_dict:
        data_dict[key].sort()
    return data_dict


def make_splitters(max_chunk_size: int) -> dict[
        str, RecursiveCharacterTextSplitter]:
    nb = max_chunk_size // 10
    py_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=max_chunk_size,
        chunk_overlap=nb, add_start_index=True)
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=max_chunk_size,
        chunk_overlap=nb, add_start_index=True)
    splitter_dict = {".py": py_splitter, ".md": md_splitter,
                     ".txt": md_splitter}
    return splitter_dict


def chunk_file(path: Path, splitter: RecursiveCharacterTextSplitter
               ) -> list[Chunk]:
    text = path.read_text(encoding='utf-8', errors='replace')
    docs = splitter.create_documents([text])
    chunks: list[Chunk] = []
    for doc in docs:
        start = doc.metadata["start_index"]
        if start == -1:
            continue
        end = start + len(doc.page_content)
        chunks.append(Chunk(file_path=str(path), first_character_index=start,
                            last_character_index=end, text=doc.page_content))
    return chunks


def test() -> None:
    data_dict = find_files("data/raw")
    for key in data_dict:
        le = len(data_dict[key])
        print(f"{key}:{le}")
