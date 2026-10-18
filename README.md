# prep-agent

[![CI](https://github.com/ahmetatakanacar/prep-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmetatakanacar/prep-agent/actions/workflows/ci.yml)

prep-agent prepares a CSV file for machine learning from a goal written in plain language, for example "prepare this for churn prediction, I will use XGBoost". It profiles the data, has an LLM propose a preparation plan, checks the plan with deterministic rules, runs it, and leaves behind the cleaned train and test files, a standalone script that repeats the preparation, and a report with baseline scores.

It exists because data preparation is where most leakage and quiet mistakes happen (a statistic fitted on the whole table, an identifier left in, a column that gives the answer away), and because letting an LLM write the code for it is a way to make those mistakes harder to see. Here the LLM chooses from a fixed set of operations, and code decides what is allowed.

## Demo
<!-- demo video: open this file on GitHub, click the pencil, drag prep-agent-demo-en.mp4 onto this line, and GitHub inserts the link here. Replace this whole comment with that link, on a line of its own. -->
https://github.com/user-attachments/assets/9c0bb51a-8383-4ac8-99ac-6026401239f8

A 75-second walkthrough of the web UI: the profile, the agent's questions, the plan with its split line, a first plan that the validators reject and the agent corrects, the baseline scores and the report. The demo runs in demo mode, with a scripted planner on `tests/fixtures/customers.csv` and no API calls. To try it yourself, see the Quick start below.

## Quick start

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). If `uv` is not installed:

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal afterwards so that `uv` is on the path. uv can fetch Python 3.12 itself (`uv python install 3.12`). Commands below that are not labelled work the same in PowerShell and bash.

### 1. Try it without an API key

```shell
git clone https://github.com/ahmetatakanacar/prep-agent.git
cd prep-agent
uv sync
uv run --extra web python -m app --fake
```

Open http://127.0.0.1:8000 in a browser and pick the `customers` sample. The `--fake` demo uses a scripted planner on `tests/fixtures/customers.csv`, makes no API calls and offers no other file. Stop the server with Ctrl+C.

### 2. Profile your own CSV (no key needed)

```shell
uv run prep-agent profile path/to/your.csv --target your_target_column
```

This prints the column types, missing rates and warnings, and nothing leaves your machine. See "Profile a file" below for the options.

### 3. Plan with a real LLM

Create an API key on the OpenAI platform's [API keys](https://platform.openai.com/api-keys) page, then copy the template and put the key after `OPENAI_API_KEY=`:

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

Run the commands from the folder that holds `.env`; it is read from the working directory and is git-ignored. Then check the key and run the agent, in the terminal or in the web UI:

```shell
uv run prep-agent llm-check
uv run prep-agent prepare path/to/your.csv --goal "prepare for churn prediction" --target your_target_column
uv run --extra openai --extra web python -m app
```

`llm-check` sends one small request and prints the model and token usage. In the web UI, upload your file or pick a sample; the nine benchmark datasets appear in the list after `uv run python benchmarks/download.py` (about 13 MB from OpenML), and only in this real-LLM mode.

The default model is `gpt-6.1-sol`. With the cheaper `gpt-6-luna` a run costs about $0.0017 (see [Benchmark](#benchmark); other models cost more):

```powershell
$env:PREP_AGENT_MODEL="gpt-6-luna"
```

```bash
export PREP_AGENT_MODEL=gpt-6-luna
```

### Troubleshooting

| Problem | What to do |
|---|---|
| `uv` is not found | Install it (above) and open a new terminal. |
| Python 3.12 is missing | `uv python install 3.12`, then run `uv sync` again. |
| `port 8000 is already in use; pick another with --port` | Another program uses that port. Start the server on another one: `uv run --extra web python -m app --fake --port 8001`, and open that port. |
| `OPENAI_API_KEY is not set: plans will fail` when the web UI starts | The server runs without a key, but a plan will not. Stop it and use `--fake`, or set the key as in the next row, then start it again. |
| `Error: OPENAI_API_KEY is not set` | Put the key in a `.env` file in the folder you run the command from, or set it in the shell. A variable already set in the shell wins over `.env`. |
| `OpenAI request failed (AuthenticationError, HTTP 401)` | The key is wrong or revoked: create a new one on the API keys page. Check the shell for an old `OPENAI_API_KEY`, which overrides `.env`. `llm-check` shows this message without running anything else. |
| The sample list has only `customers` | Demo mode (`--fake`) offers only that file. In real mode run `uv run python benchmarks/download.py`, then reload the page. |

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

- **The LLM never sees the raw data.** It gets `profile.json` only: column types, missing rates, cardinality, distribution summaries, a few sample values, and for each column how well it alone predicts the target and which earlier column it copies. Columns that look like personal data have their samples left out.
- **The LLM writes no code.** It returns a JSON plan that must match a Pydantic schema. Every step is an op from a registry (`dedupe`, `drop_columns`, `drop_missing_target`, `cast_to_category`, `split`, `impute`, `encode`, `scale`), and a step that does not match the schema or names an unknown op is rejected. pandas and scikit-learn do the work, in this project's code.
- **Leakage rules are enforced by code.** Ops are either stateless (dedupe, drop columns, cast, drop rows with a missing target) or stateful (impute, encode, scale). Stateless steps must come before the single split, stateful steps after it, and they are fitted on the train set only. The validator checks this from the kind of each op, so a plan that gets the order wrong is rejected before anything runs.
- **Open decisions become questions.** When the target, a column with many missing values or a column that could be an identifier is unclear, the planner asks you instead of guessing. If you do not give a model family (`tree`, `linear`, `any`), it asks or uses `any`.
- **The agent does not choose a model.** At the end it only scores two quick baselines per task (and a dummy model) with cross-validation on the train split. The test set is not used for that.

A plan or a result with errors is sent back to the planner for a revision, up to `--max-revisions` times (default 2); warnings are reported but do not fail a run.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/) (see [Quick start](#quick-start) to install it).

```shell
git clone https://github.com/ahmetatakanacar/prep-agent.git
cd prep-agent
uv sync --extra openai
uv run prep-agent --help
```

The commands that do not call a model (`profile` and `run`) work without the `openai` extra.

### LLM provider

The LLM layer is a small `LLMClient` interface with one adapter, OpenAI's (Responses API, structured outputs). Put your key in the environment or in a `.env` file in the working directory (`.env` is git-ignored; [.env.example](.env.example) is the template, see the [Quick start](#3-plan-with-a-real-llm)):

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

```bash
export PREP_AGENT_MODEL=gpt-6-luna
uv run prep-agent llm-check
```

`llm-check` sends one small request and prints the model and token usage; it exits 1 with a message if the key is missing or the request fails. `PREP_AGENT_PROVIDER` selects the provider; only `openai` exists.

## Usage

### Profile a file

```shell
uv run prep-agent profile tests/fixtures/customers.csv --target churned --out runs/profile.json
```

Prints a summary table; `--out` saves the full profile as JSON. Columns that look like personal data (email addresses, phone numbers, or names such as `email` or `dogum_tarihi` whose values are mostly distinct, or a name of a person such as `ad_soyad`, `soyad`, `first_name` or `fullname` whatever its values; names are compared without Turkish letters, so `Adı` and `doğum_tarihi` match too; a numeric column only by an identifier name such as `phone`, `tckn` or `iban`, so `ad_spend` keeps its statistics) are flagged `possible_pii` and their sample values and statistics are left out. `--no-samples` leaves sample values out for every column.

The profile also relates columns to each other. With a target, every numeric, categorical or binary column gets `target_association` (0 to 1): how much of the possible improvement over the trivial guess a shallow model that sees only that column reaches on rows it was not fitted on (out-of-fold R² for a numeric target, rescaled balanced accuracy for classes; a decision tree for a numeric column, a lookup table for a category). A column at 0.9 or more is flagged `suspected_target_leakage`: it nearly determines the target by itself. From 0.6 up to that it is flagged `high_target_association`. Neither flag says the column leaks: a plain fact known before the outcome (the floor area of a house, the odor of a mushroom) can predict as well as a column that exists only because of the outcome, and what tells them apart is what the column means. A column with more than 5% missing values also gets `missing_association`, the same score for the marker "is missing", judged by the same two thresholds, and `leakage_via` (`value`, `missingness` or `both`) says which one flagged it. At most 20,000 rows (a fixed sample) are used and there are no scores under 100 rows. A column that copies an earlier one, equal, scaled and shifted (a unit conversion, correlation of 0.9999 or more, or 0.99 or more when it was rounded and every value lies within one smallest step of the line) or relabelled one-to-one (`education` and `education-num`), gets `duplicate_of` with the earlier column's name and the warning `duplicate_column`. A column of unique UUIDs or hexadecimal digests is `id_like`. The planner decides a marked column by what it means, not by its score: a name that plainly says a property known before the outcome is kept with a stated reason, one that plainly says information from after the outcome is dropped (with the drop recommended when it asks), and one whose meaning it cannot tell, an anonymous name included, is put to the user. When it may not ask it drops an unclear `suspected_target_leakage` column and keeps an unclear `high_target_association` one, which the validator then reports as a warning (`high_association_kept`); that trade-off is written in the prompt template. It drops the copy of a duplicate. When you do not name the target, the profile has no relation to it while the planner picks it; once the target is known (you answered the planner's question with a column, or its plan names one) the profile is made again around it, and if that marks a column the planner is called once more with it and with its own plan (an extra call, shown as `resolve` in the trace). The thresholds and their reasons are in `ProfilerConfig`. The score is computed on all rows before the split, which lets labels of future test rows influence which columns are dropped; the effect is small (see the module docstring of `profiler.py`) but not zero. CSV files are read with pandas' usual missing-value strings, except the text `None`, which is kept as a category because in many datasets it means that a feature is absent.

How a file is read is found, not assumed, and the result is written to `profile.json` and `run.json` (`csv`). The encoding is UTF-8 (a byte order mark is fine); a file that is not valid UTF-8 is read as cp1254, the Turkish Windows code page, and a warning says so. The separator is whichever of `,` `;` tab and `|` `csv.Sniffer` finds in the first few KB, and `,` if it finds none. The decimal mark is `,` for a `;` file and `.` otherwise. `profile`, `run` and `prepare` take `--sep`, `--decimal` and `--encoding` to replace what was found (`--sep "\t"` is a tab); give them when a guess is wrong, for example `--decimal .` for a `;` file with decimal points. The generated `pipeline.py` reads its input with the same settings.

### Run a plan you wrote

```shell
uv run prep-agent run tests/fixtures/customers.csv --plan tests/fixtures/customers_plan.json --out-dir runs
```

The plan is validated first; if it has errors nothing runs. `--dry-run` only validates, `--no-baseline` skips the baseline scores. The exit code is 1 if the plan has errors or the result fails validation.

### Let the LLM plan

```shell
uv run prep-agent prepare tests/fixtures/customers.csv --goal "prepare for churn prediction, I will use XGBoost" --target churned --model-family tree
```

Questions are asked in the terminal with numbered options; the recommended one is marked and Enter picks it, and some questions allow an answer of your own as the last option. For unattended use:

```shell
uv run prep-agent prepare tests/fixtures/customers.csv --goal "prepare for churn prediction" --target churned --non-interactive
```

`--non-interactive` answers every question with the recommended option. `--model-family` is enforced: a plan for another family is sent back. If errors remain after the revisions, the command exits 1 and still saves the run with status `failed`.

### What a run leaves behind

Every run writes `runs/<run_id>/`. A real one, without the data files, is in [docs/example-run/](docs/example-run/):

| File | Content |
|---|---|
| `report.md` | goal, input, row and column counts before and after, questions and answers, every step with its reason, validator issues, baseline scores, leakage notes, how to reproduce. Built from the run's data, not written by an LLM. Failed runs get one too. |
| `plan.json` | the validated plan |
| `profile.json` | the profile the planner saw, and the CSV settings the file was read with |
| `train.csv`, `test.csv` | the prepared data |
| `pipeline.py` | the whole preparation as a standalone script (successful runs only) |
| `pipeline.joblib` | the fitted impute, encode and scale steps only; not applicable to a raw file |
| `baseline.json` | cross-validation scores of the baseline models |
| `issues.json`, `run.json` | validator issues; status, CSV settings, goal, model, token use, the language the planner named, error |
| `llm_trace.json`, `qa.json` | `prepare` only: every model call (purpose, model, token usage, issue codes, answer; no prompts, no data) and your answers |

`pipeline.py` needs only pandas and scikit-learn; prep-agent does not have to be installed. It repeats every step on a raw CSV file and gives the same `train.csv` and `test.csv` as the run folder, when the pandas and scikit-learn versions match:

```shell
python runs/<run_id>/pipeline.py tests/fixtures/customers.csv --out-dir out
```

It stops if the output files exist (`--force` overwrites) and warns if the input file differs from the one the run was made from (its sha256 is in the header, together with the target, the library versions and the run id). The reason of every step stands above its code as a comment.

### Baseline

On a successful run the baseline scores a dummy model, a logistic regression and a random forest (classification: accuracy, macro F1, ROC AUC for a binary target) or a dummy model, ridge and a random forest (regression: R², RMSE, MAE) with 5-fold cross-validation on the train split. The stateful steps are fitted again inside every fold. The test set is not used. With fewer than 100 train rows the report warns that the scores are not reliable. A model that reaches its iteration limit in a fold keeps its scores and is marked "did not converge" in the terminal table, in the report, in the web UI and in `baseline.json` (the `note` field), instead of printing a scikit-learn warning in the terminal.

## Web UI (local)

`app/` is a small web interface for trying prep-agent on your own machine: upload a CSV or pick a sample, write the goal, answer the agent's questions on the page, and read the plan with the split line, the validators' notes, how a rejected plan was corrected, the baseline scores and report.md, then download the whole run folder as a zip. An uploaded file is read with the settings found automatically (see "Profile a file"), and the profile heading names the separator and the encoding. It shows the LLM calls and token counts of a run, not what they cost. The page is plain HTML, CSS and JavaScript (no build step), in English and Turkish.

```shell
# demo: a scripted planner on tests/fixtures/customers.csv, no API calls
uv run --extra web python -m app --fake

# real LLM (needs the openai extra and a key, see above)
uv run --extra openai --extra web python -m app
```

It listens on 127.0.0.1 only, at http://127.0.0.1:8000 (`--port` changes it) and is not meant to be deployed. Sessions live in memory and their files in a temporary folder that is removed when the session is deleted or the server stops. Uploads are limited to 20 MB (`--max-upload-mb`).

## Benchmark

`benchmarks/` runs the agent on nine OpenML datasets and compares it with a hand-reviewed reference plan and a rule-based naive plan, all scored by the same baseline. In the final run (gpt-6-luna, 3 repeats per dataset) all 27 agent runs ended with a valid executed plan and none kept a column that gives the target away, which the validator's rules cannot see and which makes the naive plan look far better than it is. The agent's scores were within noise of the reference plans on 8 of 9 datasets and slightly lower on bike-sharing-demand, and careful preparation changed the score little compared with sensible defaults; the main difference was leakage. A run costs about $0.0017 with gpt-6-luna (prices from `benchmarks/prices.yaml`, which can change); the results come from one model and cross-validation on the train split, and [docs/benchmark.md](docs/benchmark.md) has the method, tables, limitations and how to repeat it.

## Validation rules

The checks that decide what a plan or a result may do live in [src/prep_agent/validators.py](src/prep_agent/validators.py) (plan, before anything runs) and [src/prep_agent/result_validator.py](src/prep_agent/result_validator.py) (result, after the executor). Errors fail the check and go back to the planner; warnings are reported in `report.md` and `issues.json`.

| Check | Errors | Warnings |
|---|---|---|
| Order and leakage | `stateless_after_split`, `stateful_before_split`, `invalid_split_count` (exactly one split) | `train_test_overlap` (the same raw record in train and test), `possible_target_leakage` (numeric feature with a correlation above 0.98 with a numeric target) |
| Target and columns | `target_mismatch`, `target_not_in_profile`, `target_constant` (the target has one value or none at all; a target given with `--target` is refused before any LLM call), `target_in_step`, `unknown_column`, `model_family_mismatch`, `missing_target_at_split` (the target still has missing values at the split: add `drop_missing_target` before it), `task_type_mismatch` (a regression plan for a target that is not numeric) | `task_type_mismatch` (a classification plan for a numeric target with more distinct values than the profiler's `regression_min_unique`) |
| Column types | `impute_non_numeric`, `fill_value_type_mismatch`, `scale_non_numeric`, `encode_numeric`, `ordinal_with_missing` | `cast_non_numeric`, `onehot_before_impute`, `ordinal_for_linear`, `scale_for_tree` |
| Split | `stratify_regression` | `stratify_disabled`, `class_distribution_shift` |
| Steps that do nothing | | `repeated_op`, `impute_without_missing`, `dedupe_without_duplicates`, `drop_missing_target_without_missing` |
| Run on the data | `execution_failed` (a step of a valid plan failed on the data; the agent sends the failure to the planner as an error) | |
| End state of the plan or result | `empty_split`, `column_mismatch`, `target_altered`, `unencoded_column_remains` (a feature that is still not numeric: encode it or drop it) | `nan_in_output`, `non_numeric_output`, `missing_values_remain`, `uninformative_column_kept`, `personal_data_kept` (a column flagged `possible_pii` is still in the data, also as one-hot columns: drop it with `drop_columns`), `suspected_leakage_kept` and `high_association_kept` (a column marked `suspected_target_leakage` or `high_target_association` is still in the data: neither mark proves a leak, so this is a warning for the report, never an error) |

## Known limitations

- The op registry cannot express everything a good plan needs. Found while reviewing the benchmark's reference plans (`benchmarks/reference/REVIEW.md`): turning special values into missing (a zero that stands for "not measured", `pdays = -1`, a top-coded 99999), ordinal encoding in a given order (quality grades, ordered scales), converting text to numbers (a number stored as text such as `TotalCharges`), and extracting features from text (the title in a passenger name). The planner can only drop or one-hot encode those columns.
- There is no date parsing: datetime and free-text columns can only be dropped.
- The leakage signal has blind spots. It cannot flag a categorical column of only a few levels cut from a continuous target (four levels explain at most about 88% of a bell-shaped target, under the 0.9 mark), a column whose link to the target runs only one way (titanic's `body` has a value only for passengers who died, but most who died have none, so the marker scores 0.00), or a leak that is noisy enough to score under 0.9 (the salary a person was paid in the year being predicted scores 0.80); the score is still in the profile for the planner to weigh. A duplicate is only found when the missing values are in the same rows, and only for equal, linearly related or one-to-one columns, not for a log or other nonlinear transform.
- Personal names are recognised by column name, not by value: a column of full names called `c07` is not flagged `possible_pii`. Email addresses and phone numbers are recognised by value.
- Only one LLM adapter exists, OpenAI's.
- Tabular data only: CSV files that fit in memory, classification and regression. No text or NLP data sets.
- A very long column name (about 80 characters or more) can make a line of the generated `pipeline.py` longer than the lint limit, so `ruff check` may report E501 for it. The script still runs.
- PII detection relies on a list of column-name parts and on email and phone value patterns. Email addresses or phone numbers inside free text are not detected.
- The benchmark covers nine datasets and one model (gpt-6-luna, 3 repeats); see its limitations.

## Roadmap

Possible next steps, roughly in the order they would change the results:

- Ops for the gaps above: special values to missing, ordinal encoding in a given order, text to number, and a few text features.
- Date parsing, so that a date column can become year, month and weekday instead of being dropped.
- A localised report (EN/TR) following the interface language.
- More LLM adapters behind the existing `LLMClient` interface.
- More benchmark datasets and models.

Out of scope: model selection or AutoML, a database, multi-user features.

## Development

```shell
uv sync --locked --extra web
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests never call the real API; they use a fake LLM. The few tests that do are marked `llm` and skipped unless you opt in. They need `OPENAI_API_KEY` and cost a few cents:

```powershell
$env:PREP_AGENT_RUN_LLM_TESTS="1"; uv run pytest -m llm -v -s
```

```bash
PREP_AGENT_RUN_LLM_TESTS=1 uv run pytest -m llm -v -s
```

The tests that need the downloaded benchmark datasets (`uv run python benchmarks/download.py`) are skipped when the files are missing. CI runs ruff and pytest on Ubuntu and Windows, with the `web` extra installed so that the web UI tests run (without it they are skipped).

## License

MIT, see [LICENSE](LICENSE).
