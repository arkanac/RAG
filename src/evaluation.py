from src.models import MinimalSource

IOU_THRESHOLD = 0.05


def inter_of_union(retrieved: MinimalSource, expected: MinimalSource
                   ) -> float:
    if retrieved.file_path != expected.file_path:
        return 0.0
    intersection = max(0, min(retrieved.last_character_index,
                              expected.last_character_index) -
                       max(retrieved.first_character_index,
                           expected.first_character_index))
    len1 = retrieved.last_character_index - retrieved.first_character_index
    len2 = (expected.last_character_index -
            expected.first_character_index)
    union = len1 + len2 - intersection
    if union == 0:
        return 0.0
    return intersection / union


def recall(retrieved: list[MinimalSource], expected: list[MinimalSource],
           k: int) -> float:
    count = 0
    if not expected:
        return 0.0
    for good_answer in expected:
        for answer in retrieved[:k]:
            if inter_of_union(answer, good_answer) >= IOU_THRESHOLD:
                count += 1
                break
    total_count = len(expected)
    return count / total_count


def test_iou() -> None:
    """Check inter_of_union on known cases."""
    def src(path: str, first: int, last: int) -> MinimalSource:
        return MinimalSource(file_path=path, first_character_index=first,
                             last_character_index=last)

    cases = [
        (src("f", 0, 100), src("f", 0, 100), 1.0),
        (src("f", 0, 100), src("f", 200, 300), 0.0),
        (src("f", 0, 100), src("f", 50, 150), 1 / 3),
        (src("f", 0, 2000), src("f", 100, 200), 0.05),
        (src("f", 0, 100), src("g", 0, 100), 0.0),
    ]
    for a, b, expected in cases:
        result = inter_of_union(a, b)
        status = "OK" if abs(result - expected) < 1e-9 else "ERREUR"
        print(f"{status}: obtenu {result:.4f}, attendu {expected:.4f}")


def test_recall() -> None:
    """Check recall on known cases."""
    def src(path: str, first: int, last: int) -> MinimalSource:
        return MinimalSource(file_path=path, first_character_index=first,
                             last_character_index=last)

    expected = [src("f", 0, 100)]
    retrieved = [src("g", 0, 100), src("f", 50, 150)]
    overlapping = [src("f", 0, 100), src("f", 10, 110)]

    cases = [
        (retrieved, expected, 1, 0.0),
        (retrieved, expected, 2, 1.0),
        (overlapping, expected, 2, 1.0),
        (retrieved, [], 2, 0.0),
    ]
    for ret, exp, k, wanted in cases:
        result = recall(ret, exp, k)
        status = "OK" if abs(result - wanted) < 1e-9 else "ERREUR"
        print(f"{status}: k={k}, obtenu {result}, attendu {wanted}")


if __name__ == "__main__":
    test_iou()
    test_recall()