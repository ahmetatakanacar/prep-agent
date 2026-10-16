# prep-agent

prep-agent prepares a CSV file for machine learning from a goal written in plain language, for example "prepare this for churn prediction, I will use XGBoost". It profiles the data, has an LLM propose a preparation plan, checks the plan with deterministic rules, runs it, and leaves behind the cleaned train and test files, a standalone script that repeats the preparation, and a report with baseline scores.

It exists because data preparation is where most leakage and quiet mistakes happen (a statistic fitted on the whole table, an identifier left in, a column that gives the answer away), and because letting an LLM write the code for it is a way to make those mistakes harder to see. Here the LLM chooses from a fixed set of operations, and code decides what is allowed.

## How it works

```
CSV + goal
   |
   v
profile ---------> profile.json (types, missing rates, cardinality, summaries, a few samples)
   |
   v
planner (LLM) <--> ask_user: questions to you when a decision is open
   |  JSON plan
   v
plan validation ---- errors: the plan goes back to the planner
   |
   v
executor: stateless steps -> split -> stateful steps (fitted on train only)
   |
   v
result validation -- errors: the plan goes back to the planner
   |
   v
baseline: cross-validation on the train split
   |
   v
report.md, train.csv, test.csv, pipeline.py, ...
```

- **The LLM never sees the raw data.** It gets `profile.json` only: column types, missing rates, cardinality, distribution summaries and a few sample values. Columns that look like personal data have their samples left out.
- **The LLM writes no code.** It returns a JSON plan that must match a Pydantic schema. Every step is an op from a registry (`dedupe`, `drop_columns`, `drop_missing_target`, `cast_to_category`, `split`, `impute`, `encode`, `scale`), and a step that does not match the schema or names an unknown op is rejected. pandas and scikit-learn do the work, in this project's code.
- **Leakage rules are enforced by code.** Ops are either stateless (dedupe, drop columns, cast, drop rows with a missing target) or stateful (impute, encode, scale). Stateless steps must come before the single split, stateful steps after it, and they are fitted on the train set only. The validator checks this from the kind of each op, so the planner cannot get it wrong.
- **Open decisions become questions.** When the target, a column with many missing values or a column that could be an identifier is unclear, the planner asks you instead of guessing. If you do not give a model family (`tree`, `linear`, `any`), it asks or uses `any`.
- **The agent does not choose a model.** At the end it only scores two quick baselines per task (and a dummy model) with cross-validation on the train split. The test set is not used for that.

A plan or a result with errors is sent back to the planner for a revision, up to `--max-revisions` times (default 2); warnings are reported but do not fail a run.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/ahmetatakanacar/prep-agent.git
cd prep-agent
uv sync --extra openai
uv run prep-agent --help
```

The commands that do not call a model (`profile` and `run`) work without the `openai` extra.

### LLM provider

The LLM layer is a small `LLMClient` interface with one adapter, OpenAI's (Responses API, structured outputs). Put your key in the environment or in a `.env` file in the working directory (`.env` is git-ignored):

```
OPENAI_API_KEY=sk-...
```

The default model is `gpt-6.1-sol`. Set `PREP_AGENT_MODEL` to choose another:

| Model | Use |
|---|---|
| `gpt-6-astra` | flagship, for the most demanding work |
| `gpt-6.1-sol` | default, close to the flagship at a lower cost |
| `gpt-6-luna` | most efficient, for focused high-volume tasks |

```powershell
$env:PREP_AGENT_MODEL="gpt-6-luna"
uv run prep-agent llm-check
```

`llm-check` sends one small request and prints the model and token usage; it exits 1 with a message if the key is missing or the request fails. `PREP_AGENT_PROVIDER` selects the provider; only `openai` exists.

## Usage

### Profile a file

```powershell
uv run prep-agent profile tests/fixtures/customers.csv --target churned --out runs/profile.json
```

Prints a summary table; `--out` saves the full profile as JSON. Columns that look like personal data (email addresses, phone numbers, or names such as `email` whose values are mostly distinct; a numeric column only by an identifier name such as `phone`, `tckn` or `iban`, so `ad_spend` keeps its statistics) are flagged `possible_pii` and their sample values and statistics are left out. `--no-samples` leaves sample values out for every column. CSV files are read with pandas' usual missing-value strings, except the text `None`, which is kept as a category because in many datasets it means that a feature is absent.

### Run a plan you wrote

```powershell
uv run prep-agent run tests/fixtures/customers.csv --plan tests/fixtures/customers_plan.json --out-dir runs
```

The plan is validated first; if it has errors nothing runs. `--dry-run` only validates, `--no-baseline` skips the baseline scores. The exit code is 1 if the plan has errors or the result fails validation.

### Let the LLM plan

```powershell
uv run prep-agent prepare tests/fixtures/customers.csv --goal "prepare for churn prediction, I will use XGBoost" --target churned --model-family tree
```

Questions are asked in the terminal with numbered options; the recommended one is marked and Enter picks it, and some questions allow an answer of your own as the last option. For unattended use:

```powershell
uv run prep-agent prepare tests/fixtures/customers.csv --goal "prepare for churn prediction" --target churned --non-interactive
```

`--non-interactive` answers every question with the recommended option. `--model-family` is enforced: a plan for another family is sent back. If errors remain after the revisions, the command exits 1 and still saves the run with status `failed`.

### What a run leaves behind

Every run writes `runs/<run_id>/`:

| File | Content |
|---|---|
| `report.md` | goal, input, row and column counts before and after, questions and answers, every step with its reason, validator issues, baseline scores, leakage notes, how to reproduce. Built from the run's data, not written by an LLM. Failed runs get one too. |
| `plan.json` | the validated plan |
| `profile.json` | the profile the planner saw |
| `train.csv`, `test.csv` | the prepared data |
| `pipeline.py` | the whole preparation as a standalone script (successful runs only) |
| `pipeline.joblib` | the fitted impute, encode and scale steps only; not applicable to a raw file |
| `baseline.json` | cross-validation scores of the baseline models |
| `issues.json`, `run.json` | validator issues; status, goal, model, token use, error |
| `llm_trace.json`, `qa.json` | `prepare` only: every model call (purpose, model, token usage, issue codes, answer; no prompts, no data) and your answers |

`pipeline.py` needs only pandas and scikit-learn; prep-agent does not have to be installed. It repeats every step on a raw CSV file and gives the same `train.csv` and `test.csv` as the run folder, when the pandas and scikit-learn versions match:

```powershell
python runs/<run_id>/pipeline.py tests/fixtures/customers.csv --out-dir out
```

It stops if the output files exist (`--force` overwrites) and warns if the input file differs from the one the run was made from (its sha256 is in the header, together with the target, the library versions and the run id). The reason of every step stands above its code as a comment.

### Baseline

On a successful run the baseline scores a dummy model, a logistic regression and a random forest (classification: accuracy, macro F1, ROC AUC for a binary target) or a dummy model, ridge and a random forest (regression: R², RMSE, MAE) with 5-fold cross-validation on the train split. The stateful steps are fitted again inside every fold. The test set is not used. With fewer than 100 train rows the report warns that the scores are not reliable. A model that reaches its iteration limit in a fold keeps its scores and is marked "did not converge" in the terminal table, in the report, in the web UI and in `baseline.json` (the `note` field), instead of printing a scikit-learn warning in the terminal.

## Web UI (local)

`app/` is a small web interface for trying prep-agent on your own machine: upload a CSV or pick a sample, write the goal, answer the agent's questions on the page, and read the plan with the split line, the validators' notes, how a rejected plan was corrected, the baseline scores and report.md, then download the whole run folder as a zip. It shows the LLM calls and token counts of a run, not what they cost. The page is plain HTML, CSS and JavaScript (no build step), in English and Turkish.

```powershell
# demo: a scripted planner on tests/fixtures/customers.csv, no API calls
uv run --extra web python -m app --fake

# real LLM (needs the openai extra and a key, see above)
uv run --extra openai --extra web python -m app
```

It listens on 127.0.0.1 only, at http://127.0.0.1:8000 (`--port` changes it) and is not meant to be deployed. Sessions live in memory and their files in a temporary folder that is removed when the session is deleted or the server stops. Uploads are limited to 20 MB (`--max-upload-mb`).

## Benchmark

`benchmarks/` runs the agent on nine OpenML datasets and compares it with a hand-reviewed reference plan and a rule-based naive plan, all scored by the same baseline. In two runs (gpt-6-luna with 3 repeats, gpt-6.1-sol with 1) the agent ended with a valid executed plan in 35 of 36 runs and never kept a column that gives the target away, a kind of leakage the validator's rules cannot see and that makes the naive plan look far better than it is. On titanic the agent dropped `boat` and `body` by itself; on employee-salaries it asked the user about the 2016 pay columns in every run and recommended dropping them, and the benchmark used that recommended answer, so asking instead of guessing when a case is unclear is designed behaviour and what is measured there is whether the recommendation is right. Its scores were within noise of the reference plans on 8 of 9 datasets for both models; careful preparation changed the score little compared with sensible defaults, and the main difference was leakage. A run costs about $0.002 with gpt-6-luna and $0.02 with gpt-6.1-sol. There is one repeat of gpt-6.1-sol, nine datasets, and the two runs used different versions of the code; see [docs/benchmark.md](docs/benchmark.md) for the method, tables, limitations and how to repeat it.

## Validation rules

The checks that decide what a plan or a result may do live in [src/prep_agent/validators.py](src/prep_agent/validators.py) (plan, before anything runs) and [src/prep_agent/result_validator.py](src/prep_agent/result_validator.py) (result, after the executor). Errors fail the check and go back to the planner; warnings are reported in `report.md` and `issues.json`.

| Check | Errors | Warnings |
|---|---|---|
| Order and leakage | `stateless_after_split`, `stateful_before_split`, `invalid_split_count` (exactly one split) | `train_test_overlap` (the same raw record in train and test), `possible_target_leakage` (numeric feature with a correlation above 0.98 with a numeric target) |
| Target and columns | `target_mismatch`, `target_not_in_profile`, `target_in_step`, `unknown_column`, `model_family_mismatch`, `missing_target_at_split` (the target still has missing values at the split: add `drop_missing_target` before it), `task_type_mismatch` (a regression plan for a target that is not numeric) | `task_type_mismatch` (a classification plan for a numeric target with more distinct values than the profiler's `regression_min_unique`) |
| Column types | `impute_non_numeric`, `fill_value_type_mismatch`, `scale_non_numeric`, `encode_numeric`, `ordinal_with_missing` | `cast_non_numeric`, `onehot_before_impute`, `ordinal_for_linear`, `scale_for_tree` |
| Split | `stratify_regression` | `stratify_disabled`, `class_distribution_shift` |
| Steps that do nothing | | `repeated_op`, `impute_without_missing`, `dedupe_without_duplicates`, `drop_missing_target_without_missing` |
| Run on the data | `execution_failed` (a step of a valid plan failed on the data; the agent sends the failure to the planner as an error) | |
| End state of the plan or result | `empty_split`, `column_mismatch`, `target_altered`, `unencoded_column_remains` (a feature that is still not numeric: encode it or drop it) | `nan_in_output`, `non_numeric_output`, `missing_values_remain`, `uninformative_column_kept`, `personal_data_kept` (a column flagged `possible_pii` is still in the data, also as one-hot columns: drop it with `drop_columns`) |

## Known limitations

- The op registry cannot express everything a good plan needs. Found while reviewing the benchmark's reference plans (`benchmarks/reference/REVIEW.md`): turning special values into missing (a zero that stands for "not measured", `pdays = -1`, a top-coded 99999), ordinal encoding in a given order (quality grades, ordered scales), converting text to numbers (a number stored as text such as `TotalCharges`), and extracting features from text (the title in a passenger name). The planner can only drop or one-hot encode those columns.
- There is no date parsing: datetime and free-text columns can only be dropped.
- The profile does not detect columns that are copies of each other (`department` and `department_name`, `education` and `education-num`), so the planner has to notice them from names and samples.
- Only one LLM adapter exists, OpenAI's.
- Tabular data only: CSV files that fit in memory, classification and regression. No text or NLP data sets.
- A very long column name (about 80 characters or more) can make a line of the generated `pipeline.py` longer than the lint limit, so `ruff check` may report E501 for it. The script still runs.
- PII detection relies on a list of column-name parts and on email and phone value patterns. Email addresses or phone numbers inside free text are not detected.
- The benchmark covers nine datasets and one repeat of the more expensive model; see its limitations.

## Roadmap

Possible next steps, roughly in the order they would change the results:

- Ops for the gaps above: special values to missing, ordinal encoding in a given order, text to number, and a few text features.
- Date parsing, so that a date column can become year, month and weekday instead of being dropped.
- Detection of duplicated columns in the profile.
- A localised report (EN/TR) following the interface language.
- More LLM adapters behind the existing `LLMClient` interface.
- More benchmark datasets and repeats of the second model.

Out of scope: model selection or AutoML, a database, multi-user features.

## Development

```powershell
uv sync --locked --extra web
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests never call the real API; they use a fake LLM. The few tests that do are marked `llm` and skipped unless you opt in. They need `OPENAI_API_KEY` and cost a few cents:

```powershell
$env:PREP_AGENT_RUN_LLM_TESTS="1"; uv run pytest -m llm -v -s
```

The tests that need the downloaded benchmark datasets (`uv run python benchmarks/download.py`) are skipped when the files are missing. CI runs ruff and pytest on Ubuntu and Windows, with the `web` extra installed so that the web UI tests run (without it they are skipped).

## License

MIT, see [LICENSE](LICENSE).
