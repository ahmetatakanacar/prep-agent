from prep_agent import __version__


def test_version_flag_prints_package_version(invoke_cli):
    result = invoke_cli(["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_help_runs_without_error(invoke_cli):
    result = invoke_cli(["--help"])

    assert result.exit_code == 0
    assert "--version" in result.output
