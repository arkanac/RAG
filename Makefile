install:
	uv sync
	@if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected: installing CUDA build of torch"; \
		uv pip install --python "$$(uv run python -c 'import sys; print(sys.executable)')" \
			torch --torch-backend=auto --reinstall; \
	else \
		echo "No NVIDIA GPU: keeping CPU build of torch"; \
	fi

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run debug clean lint lint-strict