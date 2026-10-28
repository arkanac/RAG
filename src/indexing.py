
def test() -> None:
    err_count = 0
    c_list = chunk_corpus("data/raw", 2000)
    dct: dict[str, str] = {}
    for chunk in c_list:
        if chunk.file_path not in dct:
            dct[chunk.file_path] = Path(chunk.file_path).read_text(
                encoding='utf-8', errors='replace')
        text = dct[chunk.file_path]
        cnk = text[chunk.first_character_index:chunk.last_character_index]
        if cnk != chunk.text or len(cnk) > 2000:
            err_count += 1
    print(len(c_list))
    print(err_count)


if "__main__" == __name__:
    test()
