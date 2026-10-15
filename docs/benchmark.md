# Benchmark

The benchmark measures the agent on nine real datasets against two other plans for the same data. This page describes the method, the results of the two runs kept in the repository, what they do and do not show, and how to repeat them.

Short version: the agent ended with a valid, executed plan in 35 of 36 runs, never kept a column that gives the target away (including ones the validator's rules cannot see; finding (a) says how), and scored within noise of the hand-reviewed reference plan on 8 of 9 datasets for both models. It does not prepare the data much better than a sensible rule-based plan; what it adds is that it avoids leakage and still ends with a validated plan, without a person writing one.

## Method

### Datasets

Nine OpenML datasets, six classification and three regression. The registry, with the goal sentence given to the agent and the license each dataset carries, is `benchmarks/datasets.yaml`. `benchmarks/manifest.json` holds the sha256 of every downloaded file.

| dataset | OpenML id / version | rows | columns | task | what it tests |
| --- | --- | --- | --- | --- | --- |
| titanic | 40945 / 1 | 1,309 | 14 | classification | `boat` and `body` give survival away; names are personal data; 20% of ages are missing |
| credit-g | 31 / 1 | 1,000 | 21 | classification | a small, clean, mostly categorical table where the right move is to change little |
| diabetes | 37 / 1 | 768 | 9 | classification | zeros stand for missing measurements, and no op can turn them into missing values |
| telco-customer-churn | 42178 / 1 | 7,043 | 20 | classification | 26% churn; `TotalCharges` is a number stored as text that no op converts |
| adult | 1590 / 2 | 48,842 | 15 | classification | missing values in three categorical columns, a sampling weight, a categorical column that repeats a numeric one |
| bank-marketing | 46910 / 10 | 45,211 | 14 | classification | 11.7% positive class; sentinel values (`unknown`, `pdays = -1`) |
| house-prices | 42165 / 1 | 1,460 | 81 | regression | 81 columns, an `Id`, many missing values that mean "feature absent", categories stored as numbers |
| employee-salaries | 42125 / 14 | 9,228 | 13 | regression | personal data, high-cardinality categories, a date stored as text, a near-duplicate column, and `2016_gross_pay_received`, which is what was paid for the salary being predicted |
| bike-sharing-demand | 42712 / 2 | 17,379 | 13 | regression | month, hour and weekday are integers although they are categories; `temp` and `feel_temp` are almost the same column |

The column count includes the target. Two datasets have `leaky_columns` in the registry: titanic (`boat`, `body`) and employee-salaries (`2016_gross_pay_received`). A plan that keeps one of them scores higher for a reason that has nothing to do with preparation, so the summary marks that score with a dagger (†) and leaves it out of the comparison with the naive plan.

### Three kinds of plan

Every dataset is prepared by three kinds of plan. All three are validated and executed by the same code, so a score depends on the plan only.

- **Agent.** `prep-agent prepare` run non-interactively: the dataset's target and the model family `any` are given, and every question the agent asks is answered with its recommended option. The agent sees only the profile of the data.
- **Reference.** One hand-reviewed plan per dataset in `benchmarks/reference/`, described below.
- **Naive.** A rule-based plan with no judgement: drop identifier and constant columns, split, impute with the median (numeric) or the most frequent value, one-hot encode every non-numeric column. It never scales, never casts and never drops a column that merely looks suspicious.

### Scoring

Each plan is scored with the baseline of this project: a logistic regression and a random forest for classification (ridge and a random forest for regression), 5-fold cross-validation on the plan's own train split, with the stateful steps fitted again inside every fold. The test set is never used. The main metric is macro F1 for classification and R² for regression, and the tables below give it for the linear model and the random forest as `linear / forest`.

The comparisons in the summaries use the random forest's main metric. A difference is marked **≈** when it is smaller than the larger of the two plans' fold standard deviations (for an agent with repeats, the mean over its runs). A ≈ difference is not counted as better or worse: it is not evidence of a difference.

### Reference plans

The reference plans are the yardstick, so a wrong decision in one of them becomes a wrong verdict on the agent. They were made in two steps:

1. An AI assistant drafted them from each dataset's profile and a look at its columns.
2. The project author then reviewed every plan question by question. The review sheet, `benchmarks/reference/REVIEW.md`, has for each dataset the profile, the draft plan, and the numbered questions that came up (columns to keep or drop, missing values that may mean something, suspected leakage, how to encode), each with the facts, a suggestion and the decision taken. There are 35 questions. The author agreed with the suggestion in 31 and changed the plan in 4: a `dedupe` step for adult, and three changes to the employee-salaries plan. `benchmarks/reference/README.md` explains each plan.

The review also produced a list of things a good plan needs that the op registry cannot express (turning special values into missing, ordinal encoding in a given order, converting text to numbers, extracting features from text). They are listed in the README as known limitations and affect the reference, naive and agent plans alike.

**What the review did to employee-salaries.** The AI draft dropped `underfilled_job_title` because it is 88% missing. The review found that a missing value means the employee holds the title they were hired for, and that median pay differs strongly between the two groups (53,788 when set, 73,296 when missing). The reviewed plan keeps the column, fills it with `None` and one-hot encodes it with at most 10 categories. The same commit added a missing-value indicator to `2016_overtime_pay` and let `employee_position_title` keep its 100 most frequent values instead of 50. The reference score went from 0.738 / 0.738 (linear / forest) before the review to 0.812 / 0.792 after it. These two earlier runs are not part of the committed results (they are in the git-ignored `benchmarks/results/20261014-112934` and `20261014-215647` on the author's machine), and because the commit changed three things at once, the gain cannot be credited to `underfilled_job_title` alone.

The agent still did better than the reviewed reference here: Sol's forest R² is 0.825 against 0.792, a difference larger than the fold spread. Its plan does not group rare categories, while the reference does (`max_categories` 10, 50 and 100, chosen during the review): `division`, `employee_position_title` and `underfilled_job_title` are one-hot encoded with every category, which gives 1,197 columns instead of 204. The three Luna runs point the same way (see the scores in [Results](#results)): the two plans that one-hot encode every category scored 0.825 and 0.814 and have about 1,100 and 1,200 columns, and the plan that grouped every column to 50 values scored 0.724 with 192 columns. Three runs do not prove the cause, but the reference is a careful plan, not an upper bound, and its grouping was worse than the simple alternative on this dataset.

## Results

Two runs are kept in `docs/benchmark/results/`. Each folder has the `summary.md` the benchmark wrote and the `results.json` it was made from (the plans, the scores per fold and the token use of every job; the data itself is not in it).

| run | model | repeats | code |
| --- | --- | --- | --- |
| `20261015-051059-gpt-6-luna` | gpt-6-luna | 3 | before the robustness fixes |
| `20261015-090719-gpt-6.1-sol` | gpt-6.1-sol | 1 | after the robustness fixes |

The fixes between the two runs were: a planner call whose answer does not match the schema is retried once, the validator warns about `dedupe` and `drop_missing_target` steps that the profile shows to be pointless, the prompt says that `recommended` must repeat one of the options word for word, and the cost of cached input is priced separately. A difference between the two columns below is therefore a difference of model and of code, and the benchmark cannot separate them.

### Agent runs

| | gpt-6-luna | gpt-6.1-sol |
| --- | --- | --- |
| runs | 27 (9 datasets × 3) | 9 |
| runs that ended with a valid, executed plan | 26 (96%) | 9 (100%) |
| first plan had validator errors | 5 runs (19%) | 0 |
| revisions per run (mean) | 0.2 | 0.0 |
| schema retries per run (mean) | not recorded | 0.0 |
| questions per run (mean) | 0.5 | 0.8 |
| leaky columns dropped | 9 of 9 | 3 of 3 |
| against the reference plan: better / worse / within noise | 0 / 1 / 8 | 1 / 0 / 8 |
| against the naive plan: better / worse / within noise | 0 / 1 / 6 (2 excluded †) | 0 / 1 / 6 (2 excluded †) |
| tokens per run, input / output (mean) | 9,462 / 1,617 | 9,099 / 910 |
| cost per run (USD, approximate) | 0.0018 | 0.0207 |
| plan time per run (seconds, mean) | 24.6 | 26.8 |

- The five Luna runs with a first-plan error were adult #2 and employee-salaries #1 (`stateful_before_split`) and all three bike-sharing-demand runs (`encode_numeric`). All five were sent back once and ended valid.
- The one Luna failure was telco-customer-churn #2: the model answered with a `recommended` value that is not one of its options, the answer failed schema validation, and the run ended without a plan. This is what the schema retry was added for.
- Luna's cost is the total of the 26 runs that reported usage, divided by 26 ($0.0456 / 26); the failed run's usage is unknown. Luna has no cache prices in `benchmarks/prices.yaml`, so every input token is charged the full input price and its cost is an upper bound. Sol's cost uses separate prices for cached input and cache writes. Sol costs about 12 times as much per run.
- "Against the reference/naive plan" counts the nine datasets for the reference, and the seven without a dagger for the naive plan. For Luna it compares the mean over its three runs.

### Scores

Macro F1 for classification, R² for regression; `linear / forest`, mean over folds (and over the successful runs for Luna). † marks a plan that kept a leaky column.

| dataset | reference | naive | agent: gpt-6-luna | agent: gpt-6.1-sol |
| --- | --- | --- | --- | --- |
| titanic | 0.758 / 0.781 | 0.955 / 0.945 † | 0.776 / 0.779 | 0.762 / 0.769 |
| credit-g | 0.694 / 0.663 | 0.686 / 0.677 | 0.706 / 0.661 | 0.694 / 0.677 |
| diabetes | 0.752 / 0.727 | 0.753 / 0.728 | 0.752 / 0.727 | 0.752 / 0.727 |
| telco-customer-churn | 0.729 / 0.700 | 0.730 / 0.691 | 0.732 / 0.701 (2/3 ok) | 0.732 / 0.699 |
| adult | 0.785 / 0.780 | 0.766 / 0.785 | 0.782 / 0.785 | 0.782 / 0.784 |
| bank-marketing | 0.601 / 0.614 | 0.601 / 0.614 | 0.605 / 0.616 | 0.605 / 0.617 |
| house-prices | 0.812 / 0.841 | 0.813 / 0.842 | 0.812 / 0.841 | 0.811 / 0.842 |
| employee-salaries | 0.812 / 0.792 | 0.937 / 0.972 † | 0.836 / 0.788 | 0.883 / 0.825 |
| bike-sharing-demand | 0.683 / 0.919 | 0.396 / 0.941 | 0.686 / 0.915 | 0.686 / 0.915 |

The three Luna runs on employee-salaries, scored separately (`linear / forest`): 0.883 / 0.825, 0.755 / 0.724 and 0.871 / 0.814. Sol's single run scored 0.883 / 0.825, the same as Luna's first run (1,197 columns against 1,198).

### Leakage

How many of the columns that give the target away the plan still has at its end. An agent cell adds up its runs.

| dataset | leaky columns | reference | naive | agent: gpt-6-luna | agent: gpt-6.1-sol |
| --- | --- | --- | --- | --- | --- |
| titanic | `boat`, `body` | 0 of 2 | 2 of 2 | 0 of 6 (3 runs) | 0 of 2 |
| employee-salaries | `2016_gross_pay_received` | 0 of 1 | 1 of 1 | 0 of 3 (3 runs) | 0 of 1 |

The naive plan's inflated scores are the cost of keeping them: titanic 0.945 against 0.781, employee-salaries 0.972 against 0.792 (random forest).

## Findings

**(a) The agent notices leakage that the rules cannot.** The validator's leakage rules are mechanical. The only one that looks at values, `possible_target_leakage`, runs on the result, warns instead of blocking, and needs a numeric target and an absolute correlation above 0.98. It cannot fire on `boat` and `body` in titanic, which are categorical, and not on `2016_gross_pay_received` in employee-salaries either, which correlates 0.89 with the target (the naive plan keeps it and gets no warning). The agent dropped the leaky columns in all eight runs on these two datasets (12 of 12 columns, both models). On titanic the agent dropped `boat` and `body` by itself; on employee-salaries it asked the user about the 2016 pay columns in every run and recommended dropping them, and the benchmark used that recommended answer, so asking instead of guessing when a case is unclear is designed behaviour and what is measured there is whether the recommendation is right. In an interactive run a person decides, and could decide differently. The agent also dropped `2016_overtime_pay` every time, a stricter reading than the reference's, which keeps it as a weaker, partial signal (review question employee-salaries-4).

**(b) For the score, careful preprocessing hardly beats sensible defaults; the difference is leakage.** Against the naive plan, the agent's random forest score is within noise on six of the seven comparable datasets for both models, worse on bike-sharing-demand, and better on none. The two large differences in the score are the leaky ones: the naive plan scores 0.945 on titanic and 0.972 on employee-salaries because it kept columns that give the answer away. Two caveats: the linear model does react to preparation (adult 0.766 naive against 0.782 agent, bike-sharing-demand 0.396 against 0.686, because the naive plan neither scales nor casts hour and month to categories), and on bike-sharing-demand the random forest does slightly better on the raw ordered numbers of the naive plan (0.941) than on the cast ones (0.915).

**(c) The validation layer catches the cheaper model's mistakes and the revision fixes them.** Five of Luna's 27 first plans had validator errors, none was shipped, and all five ended valid after one revision. Its final plans score within noise of the reference on 8 of 9 datasets (the exception is bike-sharing-demand, 0.004 lower), and their forest score is within 0.016 of Sol's on every dataset but employee-salaries. This is the point of the design: a plan that breaks an order rule or encodes a numeric column is rejected by code, and the model is told why. Two limits to the claim. Sol ran on later code, so its zero first-plan errors cannot be credited to the model alone. And Sol has one run per dataset.

**(d) On judgement calls Luna is not consistent from run to run.** The standard deviation of Luna's forest score across its three runs is 0.055 on employee-salaries, 0.011 on titanic and 0.002 or less on every other dataset. The employee-salaries runs differ in whether they keep `underfilled_job_title` and in how rare categories are grouped: two runs one-hot encode every category and score 0.825 and 0.814, one groups every column to 50 values and scores 0.724. Where the choices are less open (diabetes, adult, bike-sharing-demand) the three runs score the same to three decimals. Sol has no repeats, so there is no way to say whether it is more consistent.

**(e) The only Luna failure was a schema error, and it led to a fix.** In telco-customer-churn run 2 the model returned a question whose `recommended` value was not one of its options, so the answer did not match the plan schema and the run stopped. The planner now retries such a call once, telling the model what was wrong with its answer, and the prompt says that `recommended` must repeat an option word for word. Sol's run needed no schema retry.

## Limitations

- **One repeat of Sol.** Nothing in the Sol column says how much its runs vary. Its agreement with Luna's best run on employee-salaries may be luck.
- **Few datasets.** Nine, all tabular and small or medium, and none with usable dates or text features. They cover the problems listed in the table, not tabular data in general.
- **Most score differences are inside the noise.** The comparison is a single random forest on a single split, scored with 5-fold cross-validation, and most differences are below the fold spread (that is what ≈ means). The benchmark can show that the agent is not worse, and where plans have a real difference (leakage), but it cannot rank two reasonable plans.
- **The runs differ in more than the model.** Luna and Sol ran on different versions of the code (see above), so the two columns are not a clean model comparison.
- **The reference plans are one person's judgement.** The review is documented, but it is one reviewer, and the employee-salaries result shows that a reviewed choice can lose to a simpler one.
- **Non-interactive answers.** Questions are answered with the agent's own recommendation, so the benchmark does not measure what happens when a person answers differently.
- **Costs are approximate.** They are the token use the provider reported times the prices in `benchmarks/prices.yaml`, which are filled in by hand. Luna's cost ignores the cache and is an upper bound; the failed Luna run has no usage and is missing from its total.

## Reproduce

```powershell
uv sync --extra openai
uv run python benchmarks/download.py
uv run python benchmarks/run_benchmark.py --dry-run
uv run python benchmarks/run_benchmark.py --models gpt-6-luna --repeats 3 --out-dir benchmarks/results/luna
```

- `download.py` writes `benchmarks/datasets/<name>.csv` and compares each file with `benchmarks/manifest.json`. A hash that differs from the manifest is reported as a warning.
- `run_benchmark.py` without `--models` runs only the reference and naive plans, which calls no model. With `--models` it needs an API key (see the README) and a price for every model in `benchmarks/prices.yaml`. It lists the jobs and the most they can cost, and asks before it spends anything (`--yes` skips the question, `--dry-run` stops after the listing).
- `prices.yaml` ships with the two models above. Each entry has `input_per_million` and `output_per_million`; `cached_input_per_million` and `cache_write_per_million` are optional. These are the numbers the cost tables use, so check them against your provider's price list before you rely on a cost.
- `--datasets a,b` runs a subset. `--summarize benchmarks/results/<id>/results.json` writes `summary.md` again from an existing `results.json` without running anything.
- Results go to `benchmarks/results/<timestamp>/` (git-ignored) as `results.json` and `summary.md`. Model answers are not deterministic, so a new run will not reproduce these numbers exactly; the reference and naive plans do.
