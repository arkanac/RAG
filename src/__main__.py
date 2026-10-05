"""Entry point: uv run python -m src <command>."""
import fire

from .cli import RagCLI


def main() -> None:
    """Run the Fire CLI."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
