# Benchmark summary

- Started: 2026-10-18 12:36 UTC, finished: 2026-10-18 12:44 UTC
- Datasets: employee-salaries
- Models: gpt-6-luna
- Repeats: 3 per agent run; the reference and naive plans are deterministic and run once
- Estimated maximum cost approved (USD): gpt-6-luna 0.08

## Scores

Each cell is the linear model / the random forest, as the mean over the cross-validation folds (and over the successful repeats of an agent run).

### r2 ↑

| dataset | reference | naive | agent: gpt-6-luna |
| --- | --- | --- | --- |
| employee-salaries | 0.812 / 0.792 | 0.937 / 0.972 † | 0.930 / 0.922 † |

† the plan kept a column that gives the target away (`leaky_columns` in datasets.yaml). Its score is inflated by leakage and cannot be compared with the others.

## Leaky columns kept

How many of the columns that give the target away (`leaky_columns` in datasets.yaml) the plan still has at its end. An agent cell adds up its runs.

| dataset | leaky columns | reference | naive | agent: gpt-6-luna |
| --- | --- | --- | --- | --- |
| employee-salaries | 2016_gross_pay_received | 0 of 1 | 1 of 1 | 2 of 3 (3 runs) |

## Agent against the naive plan

Difference in the random forest's score on the main metric (macro F1 for classification, R² for regression): positive means the agent's plan scored higher. ≈ marks a difference smaller than 2 standard errors of the difference; it is not counted as better or worse. The error is taken from the differences of the plans' scores fold by fold when the results hold the fold scores (both plans then have the same folds), plus the spread between the runs of an agent; otherwise it is estimated from the standard deviations of the folds, which is wider. † on an agent's cell: a run kept a column that gives the target away, which counts as worse whatever its score; if it is the naive plan that kept one, the comparison is left out, because that score is inflated by the leak.

| dataset | metric | agent: gpt-6-luna |
| --- | --- | --- |
| employee-salaries | r2 | -0.049 † |
| better / worse / within noise |  | 0 / 1 / 0 (1 of the worse kept a leaky column †) |

## Agent against the reference plan

Difference in the random forest's score on the main metric (macro F1 for classification, R² for regression): positive means the agent's plan scored higher. ≈ marks a difference smaller than 2 standard errors of the difference; it is not counted as better or worse. The error is taken from the differences of the plans' scores fold by fold when the results hold the fold scores (both plans then have the same folds), plus the spread between the runs of an agent; otherwise it is estimated from the standard deviations of the folds, which is wider. † on an agent's cell: a run kept a column that gives the target away, which counts as worse whatever its score; if it is the naive plan that kept one, the comparison is left out, because that score is inflated by the leak.

| dataset | metric | agent: gpt-6-luna |
| --- | --- | --- |
| employee-salaries | r2 | +0.130 † |
| better / worse / within noise |  | 0 / 1 / 0 (1 of the worse kept a leaky column †) |

## Plan differences

Columns the final plan of each successful agent run treats differently from the reference plan. – means none.

| dataset | agent run | dropped by the reference only | dropped by the agent only | encoded or cast differently (reference vs agent) |
| --- | --- | --- | --- | --- |
| employee-salaries | gpt-6-luna #1 | 2016_gross_pay_received | – | division (onehot(50) vs onehot), employee_position_title (onehot(100) vs onehot), underfilled_job_title (onehot(10) vs onehot) |
| employee-salaries | gpt-6-luna #2 | 2016_gross_pay_received | – | division (onehot(50) vs onehot), employee_position_title (onehot(100) vs onehot), underfilled_job_title (onehot(10) vs onehot) |
| employee-salaries | gpt-6-luna #3 | – | – | division (onehot(50) vs onehot), employee_position_title (onehot(100) vs onehot), underfilled_job_title (onehot(10) vs onehot) |

## Agent repeats

Per dataset and model, over the repeats of the agent: how many runs succeeded, how many of the leaky columns the plans dropped (every run counted on its own; a run without a plan is left out) and the standard deviation of the random forest's score on the main metric from run to run (n/a below two scored runs).

| dataset | model | successful runs | leaky columns caught | score std across runs |
| --- | --- | --- | --- | --- |
| employee-salaries | gpt-6-luna | 3 of 3 (100%) | 1 of 3 (33%) | 0.084 |

## Agent runs

| model | runs | ok | revisions | schema retries | questions | input tokens | output tokens | cost (USD) | plan seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | 3 | 3 (100%) | 0.0 | 0.0 | 2.0 | 16607 | 2700 | 0.0059 | 31.8 |

## Most frequent issue codes

### First-plan errors of agent runs

No first-plan errors.

### Final-plan warnings of agent runs

| code | runs | share of runs |
| --- | --- | --- |
| high_association_kept | 3 | 100% |

### Warnings of reference and naive plans

| dataset | reference | naive |
| --- | --- | --- |
| employee-salaries | high_association_kept | high_association_kept, personal_data_kept |

## Failures

No failures.

## Notes

- Scores come from the baseline module: 5-fold cross-validation (fewer folds when a class is small) on each plan's own train split. The test set is not used. The spread between folds is in results.json; a difference smaller than that spread is not evidence that one plan is better.
- Plans differ in the rows they keep and the columns they produce, so a score compares what a plan hands to the same models, not the models themselves. The logistic regression may stop before it converges on unscaled features; that lowers the score of a plan without scaling and is not reported separately.
- The reference plans are hand-written and were reviewed by a human; see reference/README.md for the decisions behind them.
- The naive plan is rule-based: drop identifier and constant columns, split, median and most-frequent imputation, one-hot encoding of every non-numeric column.
- Cost is the token use the provider reported times the prices in prices.yaml; n/a means the provider reported no usage.
- Cost of gpt-6-luna: cached input and cache writes are priced separately.
