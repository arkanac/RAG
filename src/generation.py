"""Answer generation with Qwen."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from typing import Any
from .models import (MinimalAnswer, MinimalSource, StudentSearchResults,
                     StudentSearchResultsAndAnswer)

MAX_CONTEXT_SOURCES = 3
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

    def __init__(self, model_name: str = MODEL_NAME,
                 device: str | None = None) -> None:
        """Load the tokenizer and the model on GPU if available, else CPU."""
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        model_cls: Any = AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = model_cls.from_pretrained(model_name)
        self.model.to(self.device)

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 150) -> str:
        """Generate a greedy answer to the prompt."""
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
        new_tokens = output[0][inputs.input_ids.shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if isinstance(answer, list):
            answer = " ".join(answer)
        return answer.strip()


def read_source(source: MinimalSource) -> str:
    """Return the text of a source."""
    with open(source.file_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text[source.first_character_index:source.last_character_index]


def build_prompt(question: str, sources: list[MinimalSource]) -> str:
    """Build the user prompt from the numbered sources and the question."""
    context = "".join(f"\n\n[{i}] {s.file_path}\n{read_source(s)}"
                      for i, s in enumerate(sources, 1))
    return f"Context: \n{context}\n\nQuestion: {question}"


def answer_question(engine: TransformersEngine, question: str,
                    sources: list[MinimalSource]) -> str:
    """Answer a question from its first retrieved sources."""
    prompt = build_prompt(question, sources[:MAX_CONTEXT_SOURCES])
    return engine.generate(SYSTEM_PROMPT, prompt)


def generate_answers(engine: TransformersEngine,
                     student_search_results_path: str,
                     save_directory: str) -> None:
    """Answer every question of a search results file and save them."""
    with open(student_search_results_path, encoding="utf-8") as f:
        results = StudentSearchResults.model_validate(json.load(f))
    total = len(results.search_results)
    print(f"Loaded {total} questions")
    answers = [MinimalAnswer(question_id=r.question_id, question=r.question,
                             retrieved_sources=r.retrieved_sources,
                             answer=answer_question(engine, r.question,
                                                    r.retrieved_sources))
               for r in tqdm(results.search_results, desc="Answering")]
    print(f"Processed {len(answers)} of {total} questions")
    output = StudentSearchResultsAndAnswer(search_results=answers, k=results.k)
    out_dir = Path(save_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(student_search_results_path).name
    out_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved student_search_results_and_answer to {out_path}")
