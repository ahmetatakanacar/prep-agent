# Benchmark

The benchmark measures the agent on nine real datasets against two other plans for the same data. This page describes the method, the result of the run kept in the repository, what it does and does not show, and how to repeat it.

Short version, for the run of 16 October kept in this page (gpt-6-luna, 9 datasets × 3 repeats; later runs are under "Later runs"): all 27 agent runs ended with a valid, executed plan, none needed a schema retry, and no final plan had a warning. In that run the agent never kept a column that gives the target away, including ones the validator's rules cannot see; two later runs did keep one in one of three employee-salaries runs. Its random forest scored within noise of the hand-reviewed reference plan on 8 of 9 datasets and was slightly worse on bike-sharing-demand (R² 0.004 lower). It does not prepare the data much better than a sensible rule-based plan; what it adds is that it avoids leakage and still ends with a validated plan, without a person writing one.

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

The comparisons in the summary use the random forest's main metric. A difference is marked **≈** when it is smaller than the larger of the two plans' fold standard deviations (for an agent with repeats, the mean over its runs). A ≈ difference is not counted as better or worse: it is not evidence of a difference. **The tables below use this first rule.** Later runs use another one (see `benchmarks/bench_summary.py`): a difference is noise when it is smaller than two standard errors of the difference, taken fold by fold between the two plans when the results hold the fold scores (they do from 2026-10-18; the stored runs do not and use a wider estimate from the folds' standard deviations), plus the spread between the runs of an agent; and an agent run that kept a column that gives the target away counts as worse instead of being left out. `summary-new-noise-rule.md` next to each stored `results.json` holds the summary recomputed with it.

### Reference plans

The reference plans are the yardstick, so a wrong decision in one of them becomes a wrong verdict on the agent. They were made in two steps:

1. An AI assistant drafted them from each dataset's profile and a look at its columns.
2. The project author then reviewed every plan question by question. The review sheet, `benchmarks/reference/REVIEW.md`, has for each dataset the profile, the draft plan, and the numbered questions that came up (columns to keep or drop, missing values that may mean something, suspected leakage, how to encode), each with the facts, a suggestion and the decision taken. There are 35 questions. The author agreed with the suggestion in 31 and changed the plan in 4: a `dedupe` step for adult, and three changes to the employee-salaries plan. `benchmarks/reference/README.md` explains each plan.

The review also produced a list of things a good plan needs that the op registry cannot express (turning special values into missing, ordinal encoding in a given order, converting text to numbers, extracting features from text). They are listed in the README as known limitations and affect the reference, naive and agent plans alike.

**What the review did to employee-salaries.** The AI draft dropped `underfilled_job_title` because it is 88% missing. The review found that a missing value means the employee holds the title they were hired for, and that median pay differs strongly between the two groups (53,788 when set, 73,296 when missing). The reviewed plan keeps the column, fills it with `None` and one-hot encodes it with at most 10 categories. The same commit added a missing-value indicator to `2016_overtime_pay` and let `employee_position_title` keep its 100 most frequent values instead of 50. The reference score went from 0.738 / 0.738 (linear / forest) before the review to 0.812 / 0.792 after it. The run before the review is not part of the committed results (it is in the git-ignored `benchmarks/results/20261014-112934` on the author's machine), and because the commit changed three things at once, the gain cannot be credited to `underfilled_job_title` alone.

## Results

One run is kept in `docs/benchmark/results/20261016-115230-gpt-6-luna/`: the `summary.md` the benchmark wrote and the `results.json` it was made from (the plans, the scores per fold and the token use of every job; the data itself is not in it). It started on 2026-10-16 at 11:52 UTC and finished at 12:17 UTC, with gpt-6-luna, 3 repeats per dataset, and the agent's goal given in English. The reference and naive plans are deterministic and ran once. Earlier runs, made on earlier versions of the code, are no longer in the working tree; the two that were kept before are in the git history (commit `0815ab9`).

### Agent runs

| | gpt-6-luna |
| --- | --- |
| runs | 27 (9 datasets × 3) |
| runs that ended with a valid, executed plan | 27 (100%) |
| first plan had validator errors | 4 runs (15%) |
| revisions per run (mean) | 0.15 |
| schema retries per run (mean) | 0.0 |
| warnings on the final plan | 0 |
| questions per run (mean) | 1.2 |
| leaky columns dropped | 9 of 9 |
| against the reference plan: better / worse / within noise | 0 / 1 / 8 |
| against the naive plan: better / worse / within noise | 0 / 1 / 6 (2 excluded †) |
| tokens per run, input / output (mean) | 13,632 / 1,921 |
| cost per run (USD, mean) | 0.0017 |
| plan time per run (seconds, mean) | 25.5 |

"Against the reference/naive plan" counts the nine datasets for the reference and the seven without a dagger for the naive plan, and compares the mean over the three runs.

### Scores

Macro F1 for classification, R² for regression; `linear / forest`, mean over folds (and over the three runs for the agent). † marks a plan that kept a leaky column.

| dataset | reference | naive | agent: gpt-6-luna |
| --- | --- | --- | --- |
| titanic | 0.758 / 0.781 | 0.955 / 0.945 † | 0.769 / 0.779 |
| credit-g | 0.694 / 0.663 | 0.686 / 0.677 | 0.706 / 0.667 |
| diabetes | 0.752 / 0.727 | 0.753 / 0.728 | 0.752 / 0.727 |
| telco-customer-churn | 0.729 / 0.700 | 0.730 / 0.691 | 0.731 / 0.698 |
| adult | 0.785 / 0.780 | 0.766 / 0.785 | 0.782 / 0.785 |
| bank-marketing | 0.601 / 0.614 | 0.601 / 0.614 | 0.605 / 0.614 |
| house-prices | 0.812 / 0.841 | 0.813 / 0.842 | 0.815 / 0.843 |
| employee-salaries | 0.812 / 0.792 | 0.937 / 0.972 † | 0.859 / 0.812 |
| bike-sharing-demand | 0.683 / 0.919 | 0.396 / 0.941 | 0.686 / 0.915 |

Differences in the forest's main metric, agent minus the other plan (≈ means within noise):

| dataset | against the reference | against the naive plan |
| --- | --- | --- |
| titanic | -0.002 ≈ | -0.166 † |
| credit-g | +0.004 ≈ | -0.010 ≈ |
| diabetes | +0.000 ≈ | -0.001 ≈ |
| telco-customer-churn | -0.002 ≈ | +0.007 ≈ |
| adult | +0.004 ≈ | -0.000 ≈ |
| bank-marketing | -0.000 ≈ | +0.001 ≈ |
| house-prices | +0.002 ≈ | +0.001 ≈ |
| employee-salaries | +0.020 ≈ | -0.159 † |
| bike-sharing-demand | -0.004 | -0.026 |

### Leakage

How many of the columns that give the target away the plan still has at its end. An agent cell adds up its runs.

| dataset | leaky columns | reference | naive | agent: gpt-6-luna |
| --- | --- | --- | --- | --- |
| titanic | `boat`, `body` | 0 of 2 | 2 of 2 | 0 of 6 (3 runs) |
| employee-salaries | `2016_gross_pay_received` | 0 of 1 | 1 of 1 | 0 of 3 (3 runs) |

The naive plan's inflated scores are the cost of keeping them: titanic 0.945 against 0.781, employee-salaries 0.972 against 0.792 (random forest).

### Cost and speed

Cost is the token use the provider reported times the prices in `benchmarks/prices.yaml`, with cached input and cache writes priced separately. The 27 agent runs cost $0.0449 in total, $0.0017 per run, against a maximum of $0.76 that the benchmark estimated and asked approval for before it started. A run used on average 13,632 input tokens (8,711 of them read from the cache and 4,915 written to it) and 1,921 output tokens, in 1.8 model calls. Planning took 25.5 seconds per run on average (shortest 4.6, longest 95.8), and the whole run, including the reference and naive plans and all the baseline scoring, took about 24.5 minutes. The prices are entered by hand in `prices.yaml` and can change; check them against the provider's price list before relying on these costs.

## Findings

**(a) The agent notices leakage that the rules cannot.** The validator's leakage rules are mechanical. The only one that looks at values, `possible_target_leakage`, runs on the result, warns instead of blocking, and needs a numeric target and an absolute correlation above 0.98. It cannot fire on `boat` and `body` in titanic, which are categorical, and not on `2016_gross_pay_received` in employee-salaries either, which correlates 0.89 with the target (the naive plan keeps it and gets no warning). The agent dropped all 9 leaky columns in its 6 runs on these two datasets. In all three titanic runs it asked whether `boat` and `body` should be used to predict survival, and in all three employee-salaries runs it asked about the 2016 pay columns; the benchmark answers with the recommended option, and the prompt at that time said that for a column that may give the target away the recommended option is always the one that drops it (it has been rewritten since: a column whose name plainly says it is known before the outcome is now kept, with a reason; see "Later runs"). So what the table measures is that the agent raised the question for the right columns, not that it decided without a rule. In an interactive run a person decides, and could decide differently.

**(b) For the score, careful preprocessing hardly beats sensible defaults; the difference is leakage.** Against the naive plan, the agent's random forest score is within noise on six of the seven comparable datasets, worse on bike-sharing-demand, and better on none. The two large differences in the score are the leaky ones: the naive plan scores 0.945 on titanic and 0.972 on employee-salaries because it kept columns that give the answer away. The linear model does react to preparation (adult 0.766 naive against 0.782 agent, bike-sharing-demand 0.396 against 0.686, because the naive plan neither scales nor casts hour and month to categories).

**(c) bike-sharing-demand is the one dataset where the agent scores lower.** Its forest R² is 0.915 (runs: 0.9149, 0.9151, 0.9151) against 0.9186 for the reference and 0.9414 for the naive plan. The difference to the reference is 0.0036, and the larger fold standard deviation is 0.0035 (the reference's). The folds on this dataset vary very little, so a gap this small is just outside the noise limit and is counted as worse; it is not a large gap. The reference README says why the reference differs: "Drops `feel_temp` (correlation 0.99 with `temp`); casts `hour`, `month` and `weekday` to categories". The agent's three plans cast `month`, `hour` and `weekday` too, but they keep `feel_temp` and scale it. They also deduplicate, which removes 2 rows (13,901 train rows against the reference's 13,903), where the reference keeps them because two hours can have identical weather and demand. The results cannot say which of the two differences costs the 0.004; the benchmark did not run either one alone. Against the naive plan the gap is 0.026, and `feel_temp` cannot explain it, because the naive plan keeps `feel_temp` as well. The naive plan keeps hour, month and weekday as ordered numbers, and the random forest does better on those (0.941) than on one-hot columns (0.915); the linear model does the opposite (0.396 naive, 0.686 agent).

**(d) The validation layer catches mistakes and one revision fixes them.** In 4 of 27 runs the first plan had validator errors; none was shipped, and all four ended valid after exactly one revision. Three were `encode_numeric` (all three bike-sharing-demand runs) and one was `unencoded_column_remains` (telco-customer-churn run 1). The bike-sharing-demand case shows what the validator does. In each of the three runs the first plan put numeric columns in an `encode` step, which the validator reports twice per plan as `encode_numeric`: the column is already numeric and cannot be one-hot encoded. For a column with few distinct values its message adds that, if the values are categories (an hour, a month, a code), the plan should cast the column with `cast_to_category` before the split and keep it in the encode step. This message goes back to the planner, and in all three runs the revised plan casts `month`, `hour` and `weekday` and then encodes them. The results do not record which two columns the first plans listed or what the first telco plan left unencoded; only the error codes are kept.

**(e) Where the agent decides differently from the reference.** The summary's plan-difference table lists the columns the final plan treats differently. The ones that matter:

- *employee-salaries:* in all three runs the agent asks about `2016_overtime_pay` in the same question as `2016_gross_pay_received` and drops both. The reference keeps the overtime pay as a weaker, partial signal (review question employee-salaries-4), so this is a more cautious reading. It did not lower the score: the forest R² of the three runs is 0.826, 0.784 and 0.827 against the reference's 0.792, and the difference of the mean is within noise. Runs 1 and 3 one-hot encode `division`, `employee_position_title` and `underfilled_job_title` with every category (1,197 and 1,160 columns, against the reference's 204) and score higher than the reference. Run 2 groups rare categories at 100 values, drops `underfilled_job_title` and `gender` as well, ends with 240 columns and scores lower. Three runs that differ in several choices at once do not show which choice matters.
- *house-prices:* in all three runs the agent asks whether `MoSold`, `YrSold`, `SaleType` and `SaleCondition` will be known when the price is predicted, and drops them because they may be recorded only at or after the sale. The reference keeps them. The score is the same within noise (forest +0.002).
- *titanic:* in all three runs the agent casts `pclass` to a category and one-hot encodes it; the reference leaves it numeric. The score is the same within noise (forest -0.002).

**(f) On judgement calls the agent is not fully consistent from run to run.** The standard deviation of the forest score across the three runs is 0.024 on employee-salaries, 0.009 on credit-g, 0.005 on titanic, 0.002 on telco-customer-churn, and 0.000 on the other five datasets. The runs on employee-salaries differ in which columns they drop and how they group categories (see above).

## Later runs

The same nine datasets were run again on 18 October with later versions of the code, 3 repeats each, gpt-6-luna. All four rows are summarised with the current noise rule (`summary-new-noise-rule.md` in the older folders; the tables above use the first rule, which is why the 16 October row differs from "against the reference plan: 0 / 1 / 8" above). The story around them, and the planted-fault evaluation that prompted the changes, is in [eval.md](eval.md).

| run | code | valid plans | first plan had errors | questions/run | cost/run | leaky columns kept (of 9) | against the reference: better / worse / within noise | against the naive plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 Oct | before the profile signals | 27/27 | 4 | 1.2 | $0.0017 | 0 | 0 / 0 / 9 | 0 / 1 / 6 |
| 18 Oct | with the signals (`d584967`) | 27/27 | 3 | 0.9 | $0.0012 | 1 | 0 / 1 / 8 | 0 / 2 / 6 |
| 18 Oct | after a prompt rule on missing marks (`25418da`) | 27/27 | 3 | 1.0 | $0.0013 | 1 | 0 / 1 / 8 | 0 / 2 / 6 |
| 18 Oct, closing | `1402ac1` | 27/27 | 4 | 0.3 | $0.0011 | 0 | 2 / 0 / 7 | 0 / 1 / 6 |

Folders under `docs/benchmark/results/`: `20261016-115230-gpt-6-luna`, `20261018-081350-gpt-6-luna-signals`, `20261018-093038-gpt-6-luna-final`, `20261018-200117-gpt-6-luna-closing`. In the second and third rows the "worse" against the reference is employee-salaries, where one run kept `2016_gross_pay_received`; a run that keeps a leak counts as a loss under the new rule. In the closing run two datasets scored higher than the reference (adult +0.006, employee-salaries +0.033); on employee-salaries the plans differ from the reference in their encoding limits and in whether `employee_position_title` is kept, and the runs do not isolate which of these moves the score, so it is not evidence of better preparation (see eval.md, section 6.3). The questions per run fell; the prompt now lets the agent decide a column by its name when the name settles it (titanic and house-prices ran without a question), which is the likely reason, not a tested one. The closing run was made once, after the planted-fault evaluation, with no change to the code between it and the second correction round.

## What changed

This run (16 October) was made after a series of fixes, which were prompted by earlier runs on these same nine datasets and which this run does not test on new data:

- **Leakage rule.** When a question is about a column that may give the target away or may not be known at prediction time, the recommended option is always the one that drops it. The user can still choose otherwise.
- **Dates and text.** Datetime and free-text columns are dropped without asking, because no op can use them; the reason of the drop step says why.
- **`unencoded_column_remains` is an error.** A feature column that is still not numeric at the end of the plan used to be a warning. Now the plan is rejected and the planner revises it.
- **Language.** In earlier runs some questions came out in Spanish although the goal was in English. The planner's answer now starts with a `language` field naming the language of the goal, and the prompt says every question, option and reason is written in that language. All 33 questions asked in this run are in English. Three earlier runs had Spanish questions (a keyword search finds 3 of 16 questions in one, 11 of 25 in another and 1 of 28 in a third); they are not kept in the repository.
- **Missing-value indicators** of imputed columns are 0/1 integers.
- **Benchmark.** The cache prices of gpt-6-luna are in `prices.yaml`, and the warnings of an agent run are counted once in the summary.

## Limitations

- **One model, three repeats.** Only gpt-6-luna is in the results. Nothing here says how another model would do, and three repeats per dataset show how much a plan varies only roughly.
- **The scores are on the train split.** They are 5-fold cross-validation on each plan's own train data. The test set is not used, so the benchmark does not say how a model trained on a plan's output does on the held-out data.
- **The reference plans are an AI draft that one person reviewed.** The review is documented, but it is one reviewer, and the employee-salaries result shows that a reviewed choice can do worse than a simpler one.
- **Few datasets.** Nine, all tabular and small or medium, and none with usable dates or text features. They cover the problems listed in the table, not tabular data in general.
- **Most score differences are inside the noise.** The comparison is a single random forest on a single split, scored with 5-fold cross-validation, and most differences are below the fold spread (that is what ≈ means). The benchmark can show that the agent is not worse, and where plans have a real difference (leakage), but it cannot rank two reasonable plans.
- **The fixes were made on these datasets.** The changes listed above followed from earlier runs on the same nine datasets, so this run is not a test on unseen data. The same holds for the later runs: the changes between them followed from results on these datasets as well as from the planted-fault evaluation.
- **Non-interactive answers.** Questions are answered with the agent's own recommendation, so the benchmark does not measure what happens when a person answers differently. The planted-fault evaluation has answer modes for that (an informed user, a user who disagrees with every recommendation); see [eval.md](eval.md).
- **Costs are approximate.** They are the token use the provider reported times the prices in `benchmarks/prices.yaml`, which are filled in by hand and can change.

## Reproduce

```powershell
uv sync --extra openai
uv run python benchmarks/download.py
uv run python benchmarks/run_benchmark.py --dry-run
uv run python benchmarks/run_benchmark.py --models gpt-6-luna --repeats 3 --out-dir benchmarks/results/luna
```

- `download.py` writes `benchmarks/datasets/<name>.csv` and compares each file with `benchmarks/manifest.json`. A hash that differs from the manifest is reported as a warning.
- `run_benchmark.py` without `--models` runs only the reference and naive plans, which calls no model. With `--models` it needs an API key (see the README) and a price for every model in `benchmarks/prices.yaml`. It lists the jobs and the most they can cost, and asks before it spends anything (`--yes` skips the question, `--dry-run` stops after the listing).
- `prices.yaml` ships with prices for gpt-6-luna and gpt-6.1-sol. Each entry has `input_per_million` and `output_per_million`; `cached_input_per_million` and `cache_write_per_million` are optional. These are the numbers the cost figures use, so check them against your provider's price list before you rely on a cost.
- `--datasets a,b` runs a subset. `--summarize benchmarks/results/<id>/results.json` writes `summary.md` again from an existing `results.json` without running anything.
- Results go to `benchmarks/results/<timestamp>/` (git-ignored) as `results.json` and `summary.md`. Model answers are not deterministic, so a new run will not reproduce these numbers exactly; the reference and naive plans do.
