from src.models import (MinimalSource, RagDataset, StudentSearchResults,
                        AnsweredQuestion)
import json

IOU_THRESHOLD = 0.05
KS = (1, 3, 5, 10)


def inter_of_union(retrieved: MinimalSource, expected: MinimalSource
                   ) -> float:
    """Compute the intersection over union of two sources.
    Args:
        retrieved: Source returned by the search.
        expected: Source of the reference answer.
    Returns:
        Overlap length divided by combined length, between 0 and 1.
        0 if the sources are in different files.
    """
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
    """Compute the recall@k of one question.
    An expected source counts as found when one of the first k retrieved
    sources overlaps it with an IoU of at least IOU_THRESHOLD.
    Args:
        retrieved: Sources returned by the search, best first.
        expected: Sources of the reference answer.
        k: Number of retrieved sources taken into account.
    Returns:
        Share of expected sources found, between 0 and 1.
    """
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


def evaluate(result_path: str, dataset_path: str) -> dict[int, float]:
    """Compute the mean recall@k of a search results file.
    Args:
        result_path: StudentSearchResults file written by search_dataset.
        dataset_path: Matching dataset of answered questions.
    Returns:
        Mean recall for k = 1, 3, 5 and 10, between 0 and 1.
    Raises:
        ValueError: If the dataset has no answered question, or if no
            question of the results is in the dataset.
    """
    with open(result_path, encoding="utf-8") as f:
        results = json.load(f)
        v_result = StudentSearchResults.model_validate(results)
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
        v_data = RagDataset.model_validate(data)
    q_dict: dict[str, list[MinimalSource]] = {}
    for question in v_data.rag_questions:
        if isinstance(question, AnsweredQuestion):
            q_dict[question.question_id] = question.sources
    if not q_dict:
        raise ValueError("no answered questions found in the dataset")
    pairs = [(result.retrieved_sources, q_dict[result.question_id])
             for result in v_result.search_results
             if result.question_id in q_dict]
    if not pairs:
        raise ValueError("no question of the results matches the dataset")
    return {k: sum(recall(retrieved, expected, k)
                   for retrieved, expected in pairs) / len(pairs)
            for k in (1, 3, 5, 10)}
