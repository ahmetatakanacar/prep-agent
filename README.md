# prep-agent

Prepares a CSV dataset for machine learning from a natural-language goal: it profiles the data, has an LLM propose a structured preparation plan, validates the plan, and executes it.

**Status:** under development. The `profile`, `run`, `prepare` and `llm-check` commands work. `prepare` has the LLM plan the preparation; `run` takes a hand-written plan. The report, the standalone `pipeline.py` and the baseline scores are not implemented yet.

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

Let the LLM write the plan. `prepare` needs an LLM provider (see below). It profiles the data, asks the model for a plan, asks you when a decision is open, validates the plan, sends it back to the model if it has errors, then runs it and saves the same run folder as `run`:

```powershell
uv run prep-agent prepare tests/fixtures/customers.csv --goal "prepare for churn prediction, I will use XGBoost" --target churned --model-family tree
```

The model only sees the profile, never the raw data. Questions are asked in the terminal with numbered options; the recommended one is marked and Enter picks it. `--non-interactive` answers every question with the recommended option. `--max-revisions N` (default 2) limits how often a plan or result with errors is sent back; if errors remain, the command exits 1 and still saves the run with status `failed` (without `plan.json`, train and test files if the planner never produced a plan or the plan never ran). A question can allow a typed answer of your own (the last option); `--model-family` is enforced, so a plan for another family is sent back. Besides the usual files, the run folder has `llm_trace.json` (every model call: purpose, model, token usage, the issue codes it was asked to fix, and its answer) and `qa.json` (the questions and your answers), and `run.json` records the goal, the model, the total token usage and, for a failed run, the error.

## LLM provider

The LLM layer is a small `LLMClient` interface with an OpenAI adapter (Responses API, structured outputs). The adapter is an optional extra:

```powershell
uv sync --extra openai
```

Put `OPENAI_API_KEY` in your environment or in a `.env` file in the working directory (`.env` is git-ignored). The default model is `gpt-6.1-sol`. Set `PREP_AGENT_MODEL` to pick another one, for example:

| Model | Use |
|---|---|
| `gpt-6-astra` | flagship, for the most demanding work |
| `gpt-6.1-sol` | default, close to the flagship at a lower cost |
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
$env:PREP_AGENT_RUN_LLM_TESTS="1"; uv run pytest -m llm -v -s
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
