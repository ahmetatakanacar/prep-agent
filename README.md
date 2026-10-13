# prep-agent

Prepares a CSV dataset for machine learning from a natural-language goal: it profiles the data, has an LLM propose a structured preparation plan, validates the plan, and executes it.

**Status:** under development. The `profile`, `run` and `llm-check` commands work; `run` takes a hand-written plan, and the LLM planner is not implemented yet.

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

## Usage

Profile a CSV file and print a summary table. `--target` names the target column and `--out` saves the full profile as JSON.

```powershell
uv run prep-agent profile tests/fixtures/customers.csv --target churned --out runs/profile.json
```

Columns that look like they hold personal data (email addresses, phone numbers, or names such as `email` or `user_name`) are flagged `possible_pii` and their sample values, top values and numeric statistics are left out of the profile. Pass `--no-samples` to leave sample values out for every column.

Run a hand-written plan on a CSV file. The plan's `target` is the target column. The plan is validated first; if it has errors, nothing is run or written.

```powershell
uv run prep-agent run tests/fixtures/customers.csv --plan tests/fixtures/customers_plan.json --out-dir runs
```

This writes `runs/<run_id>/` with `profile.json`, `plan.json`, `train.csv`, `test.csv`, `pipeline.joblib`, `issues.json` and `run.json`. The exit code is 1 if the plan has errors or the result fails validation (in the second case the files are still written, with status `failed` in `run.json`); warnings do not fail a run. Pass `--dry-run` to only validate the plan.

`pipeline.joblib` contains only the fitted impute, encode and scale steps. It does not include the stateless steps or the split, so it cannot be applied to a raw file. A standalone `pipeline.py` is not generated yet.

## LLM provider

The LLM layer is a small `LLMClient` interface with an OpenAI adapter (Responses API, structured outputs). The adapter is an optional extra:

```powershell
uv sync --extra openai
```

Put `OPENAI_API_KEY` in your environment or in a `.env` file in the working directory (`.env` is git-ignored). The default model is `gpt-6-sol`. Set `PREP_AGENT_MODEL` to pick another one, for example:

| Model | Use |
|---|---|
| `gpt-6-astra` | flagship, for the most demanding work |
| `gpt-6-sol` | default, close to the flagship at a lower cost |
| `gpt-6-luna` | most efficient, for focused high-volume tasks |

```powershell
$env:PREP_AGENT_MODEL="gpt-6-astra"
```

`PREP_AGENT_PROVIDER` selects the provider (only `openai` exists). Check the setup with one small request; it prints the model and token usage, and exits 1 with a message if the key is missing or the request fails:

```powershell
uv run prep-agent llm-check
```

Tests never call the real API. The few tests that do are marked `llm` and skipped unless you opt in:

```powershell
$env:PREP_AGENT_RUN_LLM_TESTS="1"; uv run pytest -m llm -v
```

## Known limitations

- PII detection relies on a list of column-name parts and on email and phone value patterns. Email addresses or phone numbers inside free text are not detected.

## Development

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## License

MIT
