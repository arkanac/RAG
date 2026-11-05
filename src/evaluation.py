from src.models import MinimalSource


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
