from typer.testing import CliRunner

from prep_agent import __version__
from prep_agent.cli import app

runner = CliRunner()


def test_version_flag_prints_package_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_help_runs_without_error():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--version" in result.stdout
