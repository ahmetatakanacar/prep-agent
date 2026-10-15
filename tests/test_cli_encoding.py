import io
import os
import subprocess
import sys
from pathlib import Path

from prep_agent.cli import ensure_utf8_output

FIXTURES = Path(__file__).parent / "fixtures"
ARROW = "↑"


def _cp1252_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n", write_through=True)


def test_ensure_utf8_output_makes_a_cp1252_stream_accept_any_character(monkeypatch):
    out, err = _cp1252_stream(), _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    ensure_utf8_output()
    print(ARROW, file=sys.stdout)
    print(ARROW, file=sys.stderr)

    assert out.buffer.getvalue() == f"{ARROW}\n".encode()
    assert err.buffer.getvalue() == f"{ARROW}\n".encode()


def test_ensure_utf8_output_leaves_a_stream_without_reconfigure_alone(monkeypatch):
    class Plain(io.StringIO):
        encoding = "cp1252"

    stream = Plain()
    monkeypatch.setattr(sys, "stdout", stream)

    ensure_utf8_output()

    assert sys.stdout is stream


def test_run_with_output_piped_in_cp1252_does_not_crash(tmp_path):
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    command = [
        sys.executable,
        "-c",
        "from prep_agent.cli import app; app()",
        "run",
        str(FIXTURES / "customers.csv"),
        "--plan",
        str(FIXTURES / "customers_plan.json"),
        "--out-dir",
        str(tmp_path),
    ]

    result = subprocess.run(command, capture_output=True, env=env, check=False)

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert ARROW.encode() in result.stdout
