"""Loading of a local .env file."""

from pathlib import Path


def load_dotenv_file(directory: Path | None = None) -> None:
    """Load `.env` from `directory` (default: the working directory) into the environment.

    Variables that are already set win over the file. python-dotenv ships with the optional
    `openai` extra, so without it this does nothing.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=(directory or Path.cwd()) / ".env", override=False)
