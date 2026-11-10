"""Answer generation with Qwen."""
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from typing import Any
from .models import (MinimalAnswer, MinimalSource, StudentSearchResults,
                     StudentSearchResultsAndAnswer)

MAX_CONTEXT_SOURCES = 5
MODEL_NAME = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM codebase.\n"
    "Use ONLY the context below to answer.\n"
    "If the answer is not in the context, say: "
    "\"I could not find the answer in the provided sources.\"\n"
    "Answer in 2 or 3 sentences maximum.\n"
    "Never invent URLs, file names or parameters that are not in the "
    "context.\n"
    "Do not copy long code blocks: quote only the exact command, flag or "
    "name that answers the question.\n"
    "When you use a passage, cite its number, like [1]."
)


class TransformersEngine:
    """Run Qwen locally with transformers."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the tokenizer and the model once."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 256) -> str:
        """Generate an answer for the given prompt."""
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
        new_tokens = output[0][inputs.input_ids.shape[1]:]
        answer = self.tokenizer.decode(new_tokens,
                                       skip_special_tokens=True)
        if isinstance(answer, list):
            answer = " ".join(answer)
        return answer.strip()


def read_source(source: MinimalSource) -> str:
    with open(Path(source.file_path), "r", encoding="utf-8",
              errors="replace") as f:
        text = f.read()
    return text[source.first_character_index:source.last_character_index]


def build_prompt(question: str, source_lst: list[MinimalSource]) -> str:
    context: str = "Context: \n"
    for i, source in enumerate(source_lst, 1):
        context += f"\n\n[{i}] {source.file_path}\n"
        context += read_source(source)
    context += f"\n\nQuestion: {question}"
    return context


def answer_question(engine: TransformersEngine, question: str,
                    sources: list[MinimalSource]) -> str:
    """Answer a question from its retrieved sources."""
    prompt = build_prompt(question, sources[:MAX_CONTEXT_SOURCES])
    return engine.generate(SYSTEM_PROMPT, prompt)


def generate_answers(engine: TransformersEngine, student_search_results_path:
                     str, save_directory: str) -> None:
    """Answer every question of a search results file and save them."""
    with open(student_search_results_path, encoding="utf-8") as f:
        results = StudentSearchResults.model_validate(json.load(f))
    total = len(results.search_results)
    print(f"Loaded {total} questions")
    answers: list[MinimalAnswer] = []
    for result in tqdm(results.search_results, desc="Answering"):
        answer = answer_question(engine, result.question,
                                 result.retrieved_sources)
        answers.append(MinimalAnswer(question_id=result.question_id,
                                     question=result.question,
                                     retrieved_sources=result
                                     .retrieved_sources, answer=answer))
    print(f"Processed {len(answers)} of {total} questions")
    output = StudentSearchResultsAndAnswer(search_results=answers, k=results.k)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_directory) / Path(student_search_results_path).name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    print(f"Saved student_search_results_and_answer to {out_path}")
