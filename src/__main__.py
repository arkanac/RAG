"""Entry point: uv run python -m src <command>."""
import json
import sys

import fire
from pydantic import ValidationError

from .cli import RagCLI


def main() -> None:
    """Run the CLI and turn known errors into clean messages."""
    try:
        fire.Fire(RagCLI)
    except FileNotFoundError as err:
        print(f"Error: file not found: {err.filename}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as err:
        print(f"Error: invalid JSON file: {err}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as err:
        print(f"Error: unexpected file format "
              f"({err.error_count()} invalid fields)", file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
