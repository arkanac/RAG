from pathlib import Path
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from src.models import Chunk
from tqdm import tqdm


class Chunker:
    def __init__(self, max_chunk_size: int) -> None:
        self.mx_ck_size = max_chunk_size
        self.splitters: dict[str, RecursiveCharacterTextSplitter
                             ] = self._make_splitters(self.mx_ck_size)

    def find_files(self, raw_dir: str) -> dict[str, list[Path]]:
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

    def _make_splitters(self, max_chunk_size: int) -> dict[
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

    def chunk_file(self, path: Path) -> list[Chunk]:
        text = path.read_text(encoding='utf-8', errors='replace')
        splitter = self.splitters[path.suffix]
        docs = splitter.create_documents([text])
        chunks: list[Chunk] = []
        for doc in docs:
            start = doc.metadata["start_index"]
            if start == -1:
                continue
            end = start + len(doc.page_content)
            chunks.append(Chunk(file_path=str(path),
                                first_character_index=start,
                                last_character_index=end,
                                text=doc.page_content))
        return chunks

    def chunk_corpus(self, raw_dir: str) -> list[Chunk]:
        file_dict = self.find_files(raw_dir)
        chunks: list[Chunk] = []
        for extension, paths in file_dict.items():
            for path in tqdm(paths, desc=extension):
                try:
                    chunk_lst = self.chunk_file(path)
                except OSError as err:
                    print(f"{err} at path {path}")
                    continue
                chunks.extend(chunk_lst)
        return chunks
