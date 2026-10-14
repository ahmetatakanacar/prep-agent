# prep-agent

Prepares a CSV dataset for machine learning from a natural-language goal: it profiles the data, has an LLM propose a structured preparation plan, validates the plan, and executes it.

**Status:** under development. The `profile`, `run`, `prepare` and `llm-check` commands work. `prepare` has the LLM plan the preparation; `run` takes a hand-written plan. Both write a `report.md`, baseline scores and, for a successful run, a standalone `pipeline.py` into the run folder.

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

Columns that look like they hold personal data (email addresses, phone numbers, or names such as `email` or `user_name` whose values are mostly distinct) are flagged `possible_pii` and their sample values, top values and numeric statistics are left out of the profile. Pass `--no-samples` to leave sample values out for every column.

CSV files are read with pandas' usual list of missing-value strings (empty cell, `NA`, `NaN`, `null`, `N/A`, ...), except for the text `None`, which is kept as a category: in many datasets it means that a feature is absent (no pool, no veneer). The list is `NA_VALUES` in `profiler.py`, and the generated `pipeline.py` reads its input with the same one.

Run a hand-written plan on a CSV file. The plan's `target` is the target column. The plan is validated first; if it has errors, nothing is run or written.

```powershell
uv run prep-agent run tests/fixtures/customers.csv --plan tests/fixtures/customers_plan.json --out-dir runs
```

This writes `runs/<run_id>/` with `profile.json`, `plan.json`, `train.csv`, `test.csv`, `pipeline.joblib`, `pipeline.py`, `issues.json`, `report.md`, `baseline.json` and `run.json`. The exit code is 1 if the plan has errors or the result fails validation (in the second case the files are still written, with status `failed` in `run.json`); warnings do not fail a run. Pass `--dry-run` to only validate the plan and `--no-baseline` to skip the baseline scores.

`pipeline.joblib` contains only the fitted impute, encode and scale steps. It does not include the stateless steps or the split, so it cannot be applied to a raw file.

`pipeline.py` is the complete version. It needs only pandas and scikit-learn (prep-agent does not have to be installed) and repeats every step on a raw CSV file:

```powershell
python runs/<run_id>/pipeline.py tests/fixtures/customers.csv --out-dir out
```

It writes `train.csv` and `test.csv` to `--out-dir` (the current directory if left out). It stops with an error if either file already exists; `--force` overwrites them. If the input file is not the one the run was made from (its sha256 differs from the one in the header), it prints a warning to stderr and carries on. The files are identical to the files in the run folder when run on the same input with the same pandas and scikit-learn versions. The functions `apply_stateless_steps`, `split_train_test`, `build_pipeline` and `prepare` can also be imported and called one by one. The header of the script records the prep-agent version, the run id, the sha256 of the input, the target, the task type, the model family and the library versions it was generated with, and the reason of every step stands above its code as a comment. Each op writes its own code (`codegen` in its registry entry), and column names and parameters are written as literals, never as code. The script is only written for runs with status `ok`.

Let the LLM write the plan. `prepare` needs an LLM provider (see below). It profiles the data, asks the model for a plan, asks you when a decision is open, validates the plan, sends it back to the model if it has errors, then runs it and saves the same run folder as `run`:

```powershell
uv run prep-agent prepare tests/fixtures/customers.csv --goal "prepare for churn prediction, I will use XGBoost" --target churned --model-family tree
```

The model only sees the profile, never the raw data. Questions are asked in the terminal with numbered options; the recommended one is marked and Enter picks it. `--non-interactive` answers every question with the recommended option. `--max-revisions N` (default 2) limits how often a plan or result with errors is sent back; if errors remain, the command exits 1 and still saves the run with status `failed` (without `plan.json`, train and test files if the planner never produced a plan or the plan never ran). A question can allow a typed answer of your own (the last option); `--model-family` is enforced, so a plan for another family is sent back. Besides the usual files, the run folder has `llm_trace.json` (every model call: purpose, model, token usage, the issue codes it was asked to fix, and its answer) and `qa.json` (the questions and your answers), and `run.json` records the goal, the model, the total token usage and, for a failed run, the error.

## Report and baseline

Every run folder gets a `report.md`, including failed runs. It is built from the run's own data, not written by an LLM: goal, input file and short sha256, date, model and token use, row and column counts before and after (with the dropped columns), the questions and answers, the plan with each step's parameters and reason, the validator issues (or an explicit "no issues"), the baseline scores, notes on the leakage guarantees, and how to reproduce the run with `pipeline.py`. A failed run's report names the reason and has no baseline or reproduce section.

The agent does not choose a model. On a successful run it only computes reference scores with 5-fold cross-validation (stratified for classification, fewer folds if the smallest class is smaller) and writes them to `baseline.json` and the report:

- classification: a dummy model (most frequent class), logistic regression and a random forest; accuracy, macro F1 and, for a binary target, ROC AUC
- regression: a dummy model (mean), ridge and a random forest; R², RMSE and MAE

The dummy model ignores the features, so it is the floor the others should beat. Metric headers carry an arrow (↑ higher is better, ↓ lower is better), and with fewer than 100 train rows the report warns that the scores are not reliable. Cross-validation runs on the raw train split with a fresh copy of the impute, encode and scale steps in every fold, so those steps never see a fold's validation rows. The test set is not used. A model that cannot run on the prepared data (a column left with missing values or text, for example) is listed as skipped with the reason, and if the target allows no cross-validation at all the whole baseline is skipped. Pass `--no-baseline` to `run` or `prepare` to leave the baseline out; the report then says so.

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

- A very long column name (about 80 characters or more) can make a line of the generated `pipeline.py` longer than the lint limit, so `ruff check` may report E501 for it. The script still runs.
- PII detection relies on a list of column-name parts and on email and phone value patterns. Email addresses or phone numbers inside free text are not detected.

## Benchmark

The `benchmarks/` folder measures the agent on real datasets against two other plans: a hand-written reference plan and a rule-based naive plan. Every plan is validated and executed by the same code and scored with the baseline of this project (5-fold cross-validation on its own train split, the test set is not used), so a score depends on the plan only.

- `benchmarks/datasets.yaml` lists 9 OpenML datasets (6 classification, 3 regression, from 768 to 48842 rows) with the target, a goal sentence written like a user, the license OpenML reports and the difficulty each one tests. Two of them also list `leaky_columns` (titanic: `boat`, `body`; employee-salaries: `2016_gross_pay_received`), columns that give the target away: a plan that keeps one scores higher for a reason that has nothing to do with preparation, so its score is marked with a dagger in `summary.md` and left out of the comparison with the naive plan. Every job in `results.json` also records `leaky_columns_kept`, how many of those columns its plan still has at the end, and `summary.md` tabulates it per dataset and method.
- `benchmarks/reference/` holds one reference plan per dataset. They are drafts that still need a human review, except bike-sharing-demand, which has been reviewed; `benchmarks/reference/README.md` gives the status of each.
- The naive plan drops identifier and constant columns, imputes with the median (numeric) or the most frequent value, and one-hot encodes every other column.

```powershell
uv run python benchmarks/download.py
uv run python benchmarks/run_benchmark.py --dry-run
uv run python benchmarks/run_benchmark.py --models gpt-6-luna --dry-run
uv run python benchmarks/run_benchmark.py --models gpt-6-luna --repeats 3 --out-dir benchmarks/results/first
```

`download.py` writes `benchmarks/datasets/<name>.csv` (the folder is git-ignored). `benchmarks/manifest.json` is committed and holds the sha256 of every file; the downloader compares each file with it and warns when a hash differs, so a change of the data (or of how pandas writes it) does not go unnoticed. The manifest is left as it is until you pass `--update-manifest`. Files that exist are not downloaded again but are checked against the registry; a dataset whose OpenML id, name, version or shape differs from the registry is refused.

`run_benchmark.py` runs the agent non-interactively (questions are answered with the recommended option) with the dataset's target and the model family `any`, once per dataset, model and repeat, plus the reference and naive plans once per dataset. Without `--models` it runs only the reference and naive plans, which calls no model. `--datasets a,b` runs a subset, `--out-dir` sets where `results.json` and `summary.md` go (default `benchmarks/results/<timestamp>`).

Nothing is spent without a price and an approval:

- Prices are read from `benchmarks/prices.yaml`, which you fill in. The shipped file has no prices, because the project does not guess them. Each entry is named like the model passed to `--models` and has `input_per_million` and `output_per_million`:

  ```yaml
  currency: USD
  models:
    my-model:
      input_per_million: 1.0
      output_per_million: 4.0
  ```

- A model without a price is refused, in a dry run too.
- Before anything runs, the command lists the jobs and the estimated maximum use and cost of the agent runs, and asks for approval (`--yes` skips the question; declining runs nothing). The estimate is an upper bound from the size of the real prompts and the most calls the agent can make; a model that bills hidden reasoning as output can still exceed it.
- `--dry-run` stops after that listing and calls no model.

A failing job never stops the run. Each job is recorded as `ok`, `failed` (the plan or its result was rejected, with the validator codes) or `error` (anything unexpected, with the message), and `results.json` is rewritten after every job. For agent runs it records whether the run succeeded, the revisions, the questions asked, the error codes of the first plan, the warning codes of the final plan, token use, cost and time. `summary.md` has the scores per dataset and method, the difference to the naive plan, the agent statistics and the most frequent issue codes.

The tests that need the downloaded files (the reference and naive plans against the real profiles) are skipped when the files are missing.

## Development

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## License

MIT
