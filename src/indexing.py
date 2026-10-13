from pathlib import Path


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


def test():
    data_dict = find_files("data/raw")
    for key in data_dict:
        le = len(data_dict[key])
        print(f"{key}:{le}")
