"""Answer generation with Qwen."""
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


class TransformersEngine:
    """Run Qwen locally with transformers."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the tokenizer and the model once."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate an answer for the given prompt."""
        messages = [{"role": "user", "content": prompt}]
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
        answer: str = self.tokenizer.decode(new_tokens,
                                            skip_special_tokens=True)
        return answer.strip()


def test() -> None:
    """Quick manual check of the engine."""
    import time

    start = time.time()
    engine = TransformersEngine()
    print(f"Chargement : {time.time() - start:.1f} s")
    start = time.time()
    print(engine.generate("What is vLLM? Answer in one sentence."))
    print(f"Génération : {time.time() - start:.1f} s")


if __name__ == "__main__":
    test()
