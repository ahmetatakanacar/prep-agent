# Benchmark summary

- Started: 2026-10-15 09:07 UTC, finished: 2026-10-15 09:19 UTC
- Datasets: titanic, credit-g, diabetes, telco-customer-churn, adult, bank-marketing, house-prices, employee-salaries, bike-sharing-demand
- Models: gpt-6.1-sol
- Repeats: 1 per agent run; the reference and naive plans are deterministic and run once
- Estimated maximum cost approved (USD): gpt-6.1-sol 4.97

## Scores

Each cell is the linear model / the random forest, as the mean over the cross-validation folds (and over the successful repeats of an agent run).

### f1_macro ↑

| dataset | reference | naive | agent: gpt-6.1-sol |
| --- | --- | --- | --- |
| titanic | 0.758 / 0.781 | 0.955 / 0.945 † | 0.762 / 0.769 |
| credit-g | 0.694 / 0.663 | 0.686 / 0.677 | 0.694 / 0.677 |
| diabetes | 0.752 / 0.727 | 0.753 / 0.728 | 0.752 / 0.727 |
| telco-customer-churn | 0.729 / 0.700 | 0.730 / 0.691 | 0.732 / 0.699 |
| adult | 0.785 / 0.780 | 0.766 / 0.785 | 0.782 / 0.784 |
| bank-marketing | 0.601 / 0.614 | 0.601 / 0.614 | 0.605 / 0.617 |

### roc_auc ↑

| dataset | reference | naive | agent: gpt-6.1-sol |
| --- | --- | --- | --- |
| titanic | 0.834 / 0.836 | 0.981 / 0.974 † | 0.836 / 0.833 |
| credit-g | 0.788 / 0.780 | 0.790 / 0.790 | 0.788 / 0.790 |
| diabetes | 0.834 / 0.812 | 0.832 / 0.811 | 0.834 / 0.812 |
| telco-customer-churn | 0.845 / 0.812 | 0.846 / 0.812 | 0.844 / 0.815 |
| adult | 0.907 / 0.894 | 0.885 / 0.900 | 0.905 / 0.901 |
| bank-marketing | 0.742 / 0.711 | 0.742 / 0.713 | 0.749 / 0.720 |

### r2 ↑

| dataset | reference | naive | agent: gpt-6.1-sol |
| --- | --- | --- | --- |
| house-prices | 0.812 / 0.841 | 0.813 / 0.842 | 0.811 / 0.842 |
| employee-salaries | 0.812 / 0.792 | 0.937 / 0.972 † | 0.883 / 0.825 |
| bike-sharing-demand | 0.683 / 0.919 | 0.396 / 0.941 | 0.686 / 0.915 |

† the plan kept a column that gives the target away (`leaky_columns` in datasets.yaml). Its score is inflated by leakage and cannot be compared with the others.

## Leaky columns kept

How many of the columns that give the target away (`leaky_columns` in datasets.yaml) the plan still has at its end. An agent cell adds up its runs.

| dataset | leaky columns | reference | naive | agent: gpt-6.1-sol |
| --- | --- | --- | --- | --- |
| titanic | boat, body | 0 of 2 | 2 of 2 | 0 of 2 |
| employee-salaries | 2016_gross_pay_received | 0 of 1 | 1 of 1 | 0 of 1 |

## Agent against the naive plan

Difference in the random forest's score on the main metric (macro F1 for classification, R² for regression): positive means the agent's plan scored higher. ≈ marks a difference smaller than the larger of the two plans' fold standard deviations (for an agent with repeats, the mean over its runs); it is not counted as better or worse.

| dataset | metric | agent: gpt-6.1-sol |
| --- | --- | --- |
| titanic | f1_macro | -0.176 † |
| credit-g | f1_macro | +0.000 ≈ |
| diabetes | f1_macro | -0.001 ≈ |
| telco-customer-churn | f1_macro | +0.009 ≈ |
| adult | f1_macro | -0.001 ≈ |
| bank-marketing | f1_macro | +0.003 ≈ |
| house-prices | r2 | -0.000 ≈ |
| employee-salaries | r2 | -0.146 † |
| bike-sharing-demand | r2 | -0.026 |
| better / worse / within noise |  | 0 / 1 / 6 (2 excluded †) |

## Agent against the reference plan

Difference in the random forest's score on the main metric (macro F1 for classification, R² for regression): positive means the agent's plan scored higher. ≈ marks a difference smaller than the larger of the two plans' fold standard deviations (for an agent with repeats, the mean over its runs); it is not counted as better or worse.

| dataset | metric | agent: gpt-6.1-sol |
| --- | --- | --- |
| titanic | f1_macro | -0.012 ≈ |
| credit-g | f1_macro | +0.014 ≈ |
| diabetes | f1_macro | +0.000 ≈ |
| telco-customer-churn | f1_macro | -0.000 ≈ |
| adult | f1_macro | +0.003 ≈ |
| bank-marketing | f1_macro | +0.002 ≈ |
| house-prices | r2 | +0.001 ≈ |
| employee-salaries | r2 | +0.033 |
| bike-sharing-demand | r2 | -0.003 ≈ |
| better / worse / within noise |  | 1 / 0 / 8 |

## Plan differences

Columns the final plan of each successful agent run treats differently from the reference plan. – means none.

| dataset | agent run | dropped by the reference only | dropped by the agent only | encoded or cast differently (reference vs agent) |
| --- | --- | --- | --- | --- |
| titanic | gpt-6.1-sol | cabin | – | embarked (onehot vs onehot(30)), pclass (none vs cast + onehot(30)), sex (onehot vs onehot(30)) |
| credit-g | gpt-6.1-sol | – | – | – |
| diabetes | gpt-6.1-sol | – | – | – |
| telco-customer-churn | gpt-6.1-sol | – | – | – |
| adult | gpt-6.1-sol | education, fnlwgt | – | education-num (none vs cast + onehot), native-country (onehot(10) vs onehot) |
| bank-marketing | gpt-6.1-sol | – | – | – |
| house-prices | gpt-6.1-sol | Alley, GarageYrBlt, MiscFeature, PoolQC, Utilities | – | MoSold (none vs cast + onehot) |
| employee-salaries | gpt-6.1-sol | department_name | 2016_overtime_pay | division (onehot(50) vs onehot), employee_position_title (onehot(100) vs onehot), underfilled_job_title (onehot(10) vs onehot) |
| bike-sharing-demand | gpt-6.1-sol | feel_temp | – | – |

## Agent runs

| model | runs | ok | revisions | schema retries | questions | input tokens | output tokens | cost (USD) | plan seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.1-sol | 9 | 9 (100%) | 0.0 | 0.0 | 0.8 | 9099 | 910 | 0.1860 | 26.8 |

## Most frequent issue codes

### First-plan errors of agent runs

No first-plan errors.

### Final-plan warnings of agent runs

No final-plan warnings.

### Warnings of reference and naive plans

| dataset | reference | naive |
| --- | --- | --- |
| titanic | – | – |
| credit-g | – | – |
| diabetes | – | – |
| telco-customer-churn | train_test_overlap | train_test_overlap |
| adult | – | train_test_overlap |
| bank-marketing | train_test_overlap | train_test_overlap |
| house-prices | – | – |
| employee-salaries | – | – |
| bike-sharing-demand | train_test_overlap | train_test_overlap |

## Failures

No failures.

## Notes

- Scores come from the baseline module: 5-fold cross-validation (fewer folds when a class is small) on each plan's own train split. The test set is not used. The spread between folds is in results.json; a difference smaller than that spread is not evidence that one plan is better.
- Plans differ in the rows they keep and the columns they produce, so a score compares what a plan hands to the same models, not the models themselves. The logistic regression may stop before it converges on unscaled features; that lowers the score of a plan without scaling and is not reported separately.
- The reference plans are hand-written and were reviewed by a human; see reference/README.md for the decisions behind them.
- The naive plan is rule-based: drop identifier and constant columns, split, median and most-frequent imputation, one-hot encoding of every non-numeric column.
- Cost is the token use the provider reported times the prices in prices.yaml; n/a means the provider reported no usage.
- Cost of gpt-6.1-sol: cached input and cache writes are priced separately.
