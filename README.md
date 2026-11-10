*This project has been created as part of the 42 curriculum by repichan.*

# RAG against the machine

## Description

A language model only knows what it saw during training. Ask it about a specific
codebase and it will either refuse or invent a confident answer.
**Retrieval-Augmented Generation (RAG)** fixes this without retraining: before
answering, the system searches an external source for the relevant passages and
gives them to the model as context. It is an open-book exam instead of a memory test.

This project builds a complete RAG system that answers questions about the
**vLLM 0.10.1** codebase:

1. **Index** the repository: split Python and Markdown files into chunks and
   build a BM25 index.
2. **Retrieve** the most relevant chunks for a question, as exact source
   locations (`file_path` + character range).
3. **Augment** a prompt with the retrieved passages.
4. **Generate** a grounded answer with `Qwen/Qwen3-0.6B`, running locally on CPU.

Retrieval quality is measured with **recall@k**.

## Instructions

### Requirements

- Python 3.10+ and [uv](https://docs.astral.sh/uv/)
- No GPU needed: everything runs on CPU (PyTorch CPU build)
- About 2 GB of free disk space for the dependencies and the model weights

### Installation

```bash
uv sync
```

The Qwen weights (~1.5 GB) are downloaded on the first `answer` command.
On 42 machines, keep the caches out of the home directory:

```bash
export HF_HOME=~/goinfre/.cache/huggingface
export UV_CACHE_DIR=~/goinfre/.uv_cache
```

### Data layout

```
data/
├── raw/vllm-0.10.1/                     # corpus (provided)
├── processed/                           # index (built by `index`)
├── datasets/
│   ├── AnsweredQuestions/               # ground truth
│   └── UnansweredQuestions/             # questions to answer
└── output/
    ├── search_results/<Scope>/          # output of search_dataset
    └── search_results_and_answer/<Scope>/  # output of answer_dataset
```

### Commands

Every command runs as `uv run python -m src <command> [options]`.

| Command | Purpose |
|---|---|
| `index --max_chunk_size <int>` | Chunk `data/raw/` and save the index in `data/processed/` |
| `search <query> --k <int>` | Top-k sources for one question |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | Search a whole dataset, write `StudentSearchResults` JSON |
| `answer <query> --k <int>` | Answer one question from the retrieved context |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | Answer every question of a search results file |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | Compute recall@1/3/5/10 |

Optional flags: `--raw_dir` and `--processed_dir` (default `data/raw` and
`data/processed`). All paths are CLI arguments, nothing is hard-coded.

### Makefile

| Rule | Action |
|---|---|
| `make install` | `uv sync` |
| `make run ARGS="..."` | Run the CLI |
| `make debug ARGS="..."` | Run the CLI under `pdb` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |
| `make lint` / `make lint-strict` | `flake8` + `mypy` |

## Example usage

```bash
# 1. Build the index
uv run python -m src index --max_chunk_size 1900
# Ingestion complete! Indices saved under data/processed

# 2. One question
uv run python -m src search "What HTTP endpoint loads a LoRA adapter?" --k 5
uv run python -m src answer "What HTTP endpoint loads a LoRA adapter?"
# The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is
# `/v1/load_lora_adapter`.

# 3. A whole dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
# Recall@5: 0.880 (88.0%)

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Invalid inputs never crash with a traceback; they print a clear message:

```bash
uv run python -m src search "lora" --k 0
# Error: k must be at least 1, got 0
```

## System architecture

```
 data/raw ──► Chunker ──► Indexer ──► data/processed (bm25/ + chunks.json)
                                              │
 question ───────────────────────────► Retriever ──► MinimalSource list
                                              │
                          read_source + build_prompt
                                              │
                                    TransformersEngine (Qwen3-0.6B)
                                              │
                                           answer
```

| Module | Role |
|---|---|
| `src/chunking.py` | `Chunker`: finds useful files, splits them, keeps exact offsets |
| `src/indexing.py` | `Indexer`: tokenizes chunks, builds and saves the BM25 index |
| `src/retrieval.py` | `Retriever`: loads the index once, `search` and `search_dataset` |
| `src/generation.py` | Prompt building, `TransformersEngine`, `answer` / `answer_dataset` |
| `src/evaluation.py` | IoU, recall per question, recall@k over a dataset |
| `src/validation.py` | Validation of CLI parameters (`k`, `max_chunk_size`) |
| `src/models.py` | Pydantic models exchanged between the stages |
| `src/cli.py`, `src/__main__.py` | Fire CLI and central error handling |

Each stage only talks to the next through pydantic models (`Chunk`,
`MinimalSource`, `StudentSearchResults`...), and the index is loaded **once**
per command, then reused for every question.

## Chunking strategy

**File selection.** All the reference answers live in `.py`, `.md` and `.txt`
files. Going further, every code answer is under `vllm/` and every docs answer
is under `docs/` or at the repository root. The indexer therefore only keeps:

- `.py` files under `vllm-0.10.1/vllm/` (and at the root, e.g. `setup.py`),
- `.md` and `.txt` files under `vllm-0.10.1/docs/` or at the root.

`tests/`, `benchmarks/` and `examples/` are excluded: they never contain an
answer but they share a lot of vocabulary with the real sources and push them
down the ranking.

**Two splitting strategies**, both with LangChain's
`RecursiveCharacterTextSplitter.from_language`:

- **Python** (`Language.PYTHON`): splits first before `class` and `def`, then on
  blank lines, lines and words. A function stays whole when it fits.
- **Markdown / text** (`Language.MARKDOWN`): splits first before headings, then
  code blocks, paragraphs, lines and words. A section stays whole when it fits.

**Offsets.** `add_start_index=True` gives the start position of every chunk;
the end is `start + len(text)`. A test checks on the whole corpus that
`file_text[first:last] == chunk.text` and that no chunk exceeds the limit:
0 errors on all chunks.

**Size and overlap.** `chunk_size` is a maximum (default 2000, never above);
the overlap is 10 % so that an answer sitting on a boundary is not lost.

## Retrieval method

**BM25** with [bm25s](https://github.com/xhluca/bm25s) (Lucene variant). BM25
improves TF-IDF with two ideas:

- **saturation (k1)**: the 10th occurrence of a word counts less than the 2nd;
- **length normalisation (b)**: a long chunk is not favoured just because it
  contains more words;

on top of **IDF**: a rare word (`LoRARequest`) weighs much more than a common
one (`self`).

**What is indexed.** Each chunk is indexed as `file_path + " " + text`. The
words of the path (`lora`, `scheduler`, `openai`...) are strong hints of the
topic. The stored chunk text and offsets are left untouched.

**Tokenization.** `bm25s.tokenize` with lowercasing and English stopwords. The
same function is applied to the chunks and to the questions.

**Ranking.** bm25s returns chunk ids sorted by score; `chunks.json` (saved in
the same order) turns them back into `MinimalSource` locations. `k` is capped
to the number of chunks.

## Performance analysis

### Recall@k (public datasets, `k=10` searched)

Effect of the file filtering, with `max_chunk_size=1900`:

| Setup | Docs R@1 | Docs R@5 | Docs R@10 | Code R@1 | Code R@5 | Code R@10 |
|---|---|---|---|---|---|---|
| All files | 63.0 | 86.0 | 88.0 | 34.0 | 60.6 | 66.0 |
| **Filtered files** | **64.0** | **88.0** | **91.0** | **37.4** | **63.6** | **71.7** |

Required: 80 % recall@5 on docs, 50 % on code.

Effect of the chunk size on recall@5 (filtered files, public datasets):

| Chunk size | Docs R@5 | Code R@5 | Margin over thresholds* |
|---|---|---|---|
| 500 | 81.0 | 61.6 | 1.0 |
| 600 | 81.0 | 58.6 | 1.0 |
| 700 | 81.0 | 60.6 | 1.0 |
| 800 | 81.0 | 62.6 | 1.0 |
| 900 | 82.0 | 58.6 | 2.0 |
| 1000 | 85.0 | 60.6 | 5.0 |
| 1100 | 83.0 | 61.6 | 3.0 |
| 1200 | 82.0 | 63.6 | 2.0 |
| 1300 | 84.0 | 64.6 | 4.0 |
| 1400 | 82.0 | **65.7** | 2.0 |
| 1500 | 84.0 | 62.6 | 4.0 |
| 1600 | 85.0 | 63.6 | 5.0 |
| 1700 | 85.0 | 63.6 | 5.0 |
| 1800 | 86.0 | 61.6 | 6.0 |
| **1900** | **88.0** | 63.6 | **8.0** |
| 2000 (default) | 87.0 | 61.6 | 7.0 |

\* `min(docs - 80, code - 50)`: the distance to the closest threshold.

- Every size passes both thresholds once the files are filtered, but below
  900 characters docs stay just above 80 %: small chunks cut long
  documentation answers.
- Code peaks around 1300-1400: code answers are shorter, so smaller chunks
  match them more precisely.
- 1900 gives the best safety margin on both datasets; 2000 (the required
  default) is almost as good.
- The differences are small (one question = 1 point), so chunk size is a weak
  lever compared to file selection.

The `evaluate` command gives exactly the same numbers as the moulinette.

### Speed

| Step | Time |
|---|---|
| Indexing the whole corpus | ~2 s (limit: 5 min) |
| Searching 100 questions | < 0.1 s (limit: 90 s for 200) |
| Loading Qwen (cached) | ~2 s |
| Generating one answer (5 passages, CPU) | ~15-20 s |

### Answer quality

With the final prompt, most answers are short, correct and grounded
(`/v1/load_lora_adapter`, `VLLM_ATTENTION_BACKEND`, `SupportsMultiModal`,
`VllmConfig`...). When the context does not contain the answer, the model says
so instead of guessing. Remaining errors come from the 0.6B model itself:
wrong parameter picked from a correct context, and a few invented URLs.

## Design decisions

- **BM25 over TF-IDF**: saturation and length normalisation make it the
  standard for lexical search; bm25s is fast and saves/loads its index.
- **Library splitters** instead of a hand-written AST chunker: two distinct
  strategies with exact offsets, verified by a test on the full corpus.
- **Data-driven file filtering**: chosen by looking at where the reference
  answers are.
- **transformers inside the program** to run Qwen, instead of a vLLM server:
  everything installs with `uv sync`, nothing has to be launched beside it.
- **Prompt strategy**: a system message with short, explicit rules (use only
  the context, fixed sentence when the answer is missing, 2-3 sentences, no
  invented URLs, no long code blocks), then a user message with numbered
  passages and their file paths, the question last. `enable_thinking=False`
  and greedy decoding (`do_sample=False`) give fast and reproducible answers.
  Only the first 5 retrieved sources go into the prompt to fit the token
  budget and keep generation time reasonable.
- **One place for errors**: parameters are validated in `validation.py`, and
  `__main__.py` turns every known error (missing file, broken JSON, wrong
  format, invalid value, Ctrl+C) into a clean message. A script runs 26 edge
  cases and checks that none produces a traceback.

## Challenges faced

- **Running Qwen with vLLM.** The CPU nightly build of vLLM was only published
  for ARM on the day of the install, so the install failed on x86. Rather than
  depending on a moving target, generation runs with `transformers` + PyTorch
  CPU, which makes the project reproducible with `uv sync` alone.
- **Noise in retrieval.** A question about generative models was answered
  with a description of another AI assistant: a test file containing a system
  prompt ranked first. Excluding `tests/` and `benchmarks/` fixed it and raised
  recall on both datasets.
- **The IoU trap.** Some reference spans are tiny (93 characters): with 2000
  character chunks, even a chunk containing the answer is under the 0.05 IoU
  bar. This is why chunk size was measured instead of maxed out.
- **Hallucinated links and truncated answers** with the first prompt: fixed by
  stricter rules on length, links and code in the system prompt.
- **Typing with transformers.** Some transformers annotations are inconsistent
  for mypy; solved with an `Any`-typed model attribute and an explicit
  `isinstance` check on decoded text.
- **Corrupted git index** several times on school machines (home directory
  synchronisation). Committing and pushing often to two remotes meant nothing
  was ever lost.

## Resources

### Documentation and articles

- [bm25s](https://github.com/xhluca/bm25s) — the BM25 implementation used
- [LangChain text splitters](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
- [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face transformers — text generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Pinecone — Chunking strategies for LLM applications](https://www.pinecone.io/learn/chunking-strategies/)
- [Alexis Perrier — Dans le RAG, l'important c'est le chunk](https://medium.com/@alexisperrier/dans-le-rag-limportant-c-est-le-chunk-0d541b4f4bbe)
- [Datascientist.fr — Architecture RAG : ingestion, indexation, retrieval, génération](https://datascientist.fr/blog/architecture-rag-ingestion-indexation-retrieval-generation)

### How AI was used

An AI assistant (Claude) was used as a tutor throughout the project:

- **Explanations of concepts**: BM25 vs TF-IDF, embeddings, IoU, recall@k,
  tokenization, stemming, the role of PyTorch / transformers / vLLM.
- **Analysis**: reading benchmark outputs and generated answers to decide the next improvement (chunk size, file filtering, prompt rules).