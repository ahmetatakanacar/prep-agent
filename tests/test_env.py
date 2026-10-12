import builtins
import os

import pytest

from prep_agent.env import load_dotenv_file

VARIABLE = "PREP_AGENT_TEST_VAR"


@pytest.fixture(autouse=True)
def unset_variable(monkeypatch):
    # Set-then-delete makes monkeypatch remove the variable again at teardown, even though
    # load_dotenv_file puts it into os.environ behind monkeypatch's back.
    monkeypatch.setenv(VARIABLE, "placeholder")
    monkeypatch.delenv(VARIABLE)


def write_env(directory, value):
    (directory / ".env").write_text(f"{VARIABLE}={value}\n", encoding="utf-8")


def test_loads_variables_from_the_given_directory(tmp_path):
    write_env(tmp_path, "from-file")

    load_dotenv_file(tmp_path)

    assert os.environ[VARIABLE] == "from-file"


def test_defaults_to_the_working_directory(tmp_path, monkeypatch):
    write_env(tmp_path, "from-cwd")
    monkeypatch.chdir(tmp_path)

    load_dotenv_file()

    assert os.environ[VARIABLE] == "from-cwd"


def test_existing_environment_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE, "from-shell")
    write_env(tmp_path, "from-file")

    load_dotenv_file(tmp_path)

    assert os.environ[VARIABLE] == "from-shell"


def test_missing_file_is_not_an_error(tmp_path):
    load_dotenv_file(tmp_path)

    assert VARIABLE not in os.environ


def test_missing_dotenv_package_is_not_an_error(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def import_without_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("No module named 'dotenv'")
        return real_import(name, *args, **kwargs)

    write_env(tmp_path, "x")
    monkeypatch.setattr(builtins, "__import__", import_without_dotenv)

    load_dotenv_file(tmp_path)

    assert VARIABLE not in os.environ
