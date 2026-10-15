"""Settings of the local web UI."""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# The one place the upload limit is defined; --max-upload-mb overrides it.
DEFAULT_MAX_UPLOAD_MB = 20


@dataclass(frozen=True)
class Settings:
    fake: bool = False
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    # Host names the server answers to. Anything else is refused, which keeps a web page on
    # another origin from reaching the server through a DNS-rebinding trick.
    allowed_hosts: tuple[str, ...] = (HOST, "localhost")
