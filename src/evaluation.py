from src.models import (MinimalSource, RagDataset, StudentSearchResults,
                        AnsweredQuestion)
import json

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


def evaluate(result_path: str, dataset_path: str) -> None:
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
        print("No answered questions found in the dataset.")
        return
    for k in [1, 3, 5, 10]:
        recall_list: list[float] = []
        for result in v_result.search_results:
            expected = q_dict.get(result.question_id)
            if expected is None:
                continue
            recall_list.append(
                recall(result.retrieved_sources, expected, k))
        if not recall_list:
            print("No question of the results matches the dataset.")
            return
        recall_k = sum(recall_list) / len(recall_list)
        print(f"Recall@{k}: {recall_k:.3f} ({recall_k:.1%})")
