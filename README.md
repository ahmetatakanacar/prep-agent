# prep-agent

Prepares a CSV dataset for machine learning from a natural-language goal: it profiles the data, has an LLM propose a structured preparation plan, validates the plan, and executes it.

**Status:** under development. Nothing is usable yet beyond the CLI skeleton.

## Planned features

- Data profiling: the LLM only ever sees a `profile.json` summary, never the raw data.
- Plans as validated JSON: every step maps to a registered operation, and the LLM writes no free-form code.
- Leakage protection built into the validator: stateful steps (impute, encode, scale) run after the train/test split and are fit on train only.
- Reproducible output: cleaned train/test CSVs, a fitted pipeline, a standalone `pipeline.py`, and a report with baseline cross-validation scores.
- Provider-independent LLM layer, tested with a fake client instead of real API calls.

## Install from source

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run prep-agent --help
```

## Development

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## License

MIT
