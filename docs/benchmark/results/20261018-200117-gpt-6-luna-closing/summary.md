# Benchmark summary

- Started: 2026-10-18 20:01 UTC, finished: 2026-10-18 20:35 UTC
- Datasets: titanic, credit-g, diabetes, telco-customer-churn, adult, bank-marketing, house-prices, employee-salaries, bike-sharing-demand
- Models: gpt-6-luna
- Repeats: 3 per agent run; the reference and naive plans are deterministic and run once
- Estimated maximum cost approved (USD): gpt-6-luna 0.81

## Scores

Each cell is the linear model / the random forest, as the mean over the cross-validation folds (and over the successful repeats of an agent run).

### f1_macro ↑

| dataset | reference | naive | agent: gpt-6-luna |
| --- | --- | --- | --- |
| titanic | 0.758 / 0.781 | 0.955 / 0.945 † | 0.774 / 0.765 |
| credit-g | 0.694 / 0.663 | 0.686 / 0.677 | 0.706 / 0.662 |
| diabetes | 0.752 / 0.727 | 0.753 / 0.728 | 0.752 / 0.727 |
| telco-customer-churn | 0.729 / 0.700 | 0.730 / 0.691 | 0.731 / 0.700 |
| adult | 0.785 / 0.780 | 0.766 / 0.785 | 0.784 / 0.787 |
| bank-marketing | 0.601 / 0.614 | 0.601 / 0.614 | 0.605 / 0.614 |

### roc_auc ↑

| dataset | reference | naive | agent: gpt-6-luna |
| --- | --- | --- | --- |
| titanic | 0.834 / 0.836 | 0.981 / 0.974 † | 0.849 / 0.834 |
| credit-g | 0.788 / 0.780 | 0.790 / 0.790 | 0.794 / 0.780 |
| diabetes | 0.834 / 0.812 | 0.832 / 0.811 | 0.834 / 0.812 |
| telco-customer-churn | 0.845 / 0.812 | 0.846 / 0.812 | 0.844 / 0.815 |
| adult | 0.907 / 0.894 | 0.885 / 0.900 | 0.906 / 0.903 |
| bank-marketing | 0.742 / 0.711 | 0.742 / 0.713 | 0.749 / 0.720 |

### r2 ↑

| dataset | reference | naive | agent: gpt-6-luna |
| --- | --- | --- | --- |
| house-prices | 0.812 / 0.841 | 0.813 / 0.842 | 0.813 / 0.844 |
| employee-salaries | 0.812 / 0.792 | 0.937 / 0.972 † | 0.884 / 0.826 |
| bike-sharing-demand | 0.683 / 0.919 | 0.396 / 0.941 | 0.686 / 0.915 |

† the plan kept a column that gives the target away (`leaky_columns` in datasets.yaml). Its score is inflated by leakage and cannot be compared with the others.

## Leaky columns kept

How many of the columns that give the target away (`leaky_columns` in datasets.yaml) the plan still has at its end. An agent cell adds up its runs.

| dataset | leaky columns | reference | naive | agent: gpt-6-luna |
| --- | --- | --- | --- | --- |
| titanic | boat, body | 0 of 2 | 2 of 2 | 0 of 6 (3 runs) |
| employee-salaries | 2016_gross_pay_received | 0 of 1 | 1 of 1 | 0 of 3 (3 runs) |

## Agent against the naive plan

Difference in the random forest's score on the main metric (macro F1 for classification, R² for regression): positive means the agent's plan scored higher. ≈ marks a difference smaller than 2 standard errors of the difference; it is not counted as better or worse. The error is taken from the differences of the plans' scores fold by fold when the results hold the fold scores (both plans then have the same folds), plus the spread between the runs of an agent; otherwise it is estimated from the standard deviations of the folds, which is wider. † on an agent's cell: a run kept a column that gives the target away, which counts as worse whatever its score; if it is the naive plan that kept one, the comparison is left out, because that score is inflated by the leak.

| dataset | metric | agent: gpt-6-luna |
| --- | --- | --- |
| titanic | f1_macro | -0.180 † |
| credit-g | f1_macro | -0.015 ≈ |
| diabetes | f1_macro | -0.001 ≈ |
| telco-customer-churn | f1_macro | +0.009 ≈ |
| adult | f1_macro | +0.002 ≈ |
| bank-marketing | f1_macro | +0.001 ≈ |
| house-prices | r2 | +0.002 ≈ |
| employee-salaries | r2 | -0.146 † |
| bike-sharing-demand | r2 | -0.026 |
| better / worse / within noise |  | 0 / 1 / 6 (2 excluded †) |

## Agent against the reference plan

Difference in the random forest's score on the main metric (macro F1 for classification, R² for regression): positive means the agent's plan scored higher. ≈ marks a difference smaller than 2 standard errors of the difference; it is not counted as better or worse. The error is taken from the differences of the plans' scores fold by fold when the results hold the fold scores (both plans then have the same folds), plus the spread between the runs of an agent; otherwise it is estimated from the standard deviations of the folds, which is wider. † on an agent's cell: a run kept a column that gives the target away, which counts as worse whatever its score; if it is the naive plan that kept one, the comparison is left out, because that score is inflated by the leak.

| dataset | metric | agent: gpt-6-luna |
| --- | --- | --- |
| titanic | f1_macro | -0.015 ≈ |
| credit-g | f1_macro | -0.001 ≈ |
| diabetes | f1_macro | +0.000 ≈ |
| telco-customer-churn | f1_macro | +0.000 ≈ |
| adult | f1_macro | +0.006 |
| bank-marketing | f1_macro | -0.000 ≈ |
| house-prices | r2 | +0.003 ≈ |
| employee-salaries | r2 | +0.033 |
| bike-sharing-demand | r2 | -0.004 ≈ |
| better / worse / within noise |  | 2 / 0 / 7 |

## Plan differences

Columns the final plan of each successful agent run treats differently from the reference plan. – means none.

| dataset | agent run | dropped by the reference only | dropped by the agent only | encoded or cast differently (reference vs agent) |
| --- | --- | --- | --- | --- |
| titanic | gpt-6-luna #1 | cabin, home.dest, ticket | – | embarked (onehot vs onehot(50)), pclass (none vs cast + onehot(50)), sex (onehot vs onehot(50)) |
| titanic | gpt-6-luna #2 | cabin, home.dest, ticket | – | embarked (onehot vs onehot(50)), pclass (none vs cast + onehot(50)), sex (onehot vs onehot(50)) |
| titanic | gpt-6-luna #3 | cabin, home.dest, ticket | – | embarked (onehot vs onehot(50)), pclass (none vs cast + onehot(50)), sex (onehot vs onehot(50)) |
| credit-g | gpt-6-luna #1 | – | – | – |
| credit-g | gpt-6-luna #2 | – | – | installment_commitment (none vs cast + onehot), residence_since (none vs cast + onehot) |
| credit-g | gpt-6-luna #3 | – | – | installment_commitment (none vs cast + onehot), residence_since (none vs cast + onehot) |
| diabetes | gpt-6-luna #1 | – | – | – |
| diabetes | gpt-6-luna #2 | – | – | – |
| diabetes | gpt-6-luna #3 | – | – | – |
| telco-customer-churn | gpt-6-luna #1 | – | – | – |
| telco-customer-churn | gpt-6-luna #2 | TotalCharges | – | Contract (onehot vs onehot(50)), Dependents (onehot vs onehot(50)), DeviceProtection (onehot vs onehot(50)), InternetService (onehot vs onehot(50)), MultipleLines (onehot vs onehot(50)), OnlineBackup (onehot vs onehot(50)), OnlineSecurity (onehot vs onehot(50)), PaperlessBilling (onehot vs onehot(50)), Partner (onehot vs onehot(50)), PaymentMethod (onehot vs onehot(50)), PhoneService (onehot vs onehot(50)), StreamingMovies (onehot vs onehot(50)), StreamingTV (onehot vs onehot(50)), TechSupport (onehot vs onehot(50)), gender (onehot vs onehot(50)) |
| telco-customer-churn | gpt-6-luna #3 | TotalCharges | – | Contract (onehot vs onehot(20)), Dependents (onehot vs onehot(20)), DeviceProtection (onehot vs onehot(20)), InternetService (onehot vs onehot(20)), MultipleLines (onehot vs onehot(20)), OnlineBackup (onehot vs onehot(20)), OnlineSecurity (onehot vs onehot(20)), PaperlessBilling (onehot vs onehot(20)), Partner (onehot vs onehot(20)), PaymentMethod (onehot vs onehot(20)), PhoneService (onehot vs onehot(20)), SeniorCitizen (none vs cast + onehot(20)), StreamingMovies (onehot vs onehot(20)), StreamingTV (onehot vs onehot(20)), TechSupport (onehot vs onehot(20)), gender (onehot vs onehot(20)) |
| adult | gpt-6-luna #1 | education, fnlwgt | education-num | native-country (onehot(10) vs onehot) |
| adult | gpt-6-luna #2 | education, fnlwgt | education-num | native-country (onehot(10) vs onehot) |
| adult | gpt-6-luna #3 | education, fnlwgt | education-num | native-country (onehot(10) vs onehot) |
| bank-marketing | gpt-6-luna #1 | – | – | – |
| bank-marketing | gpt-6-luna #2 | – | – | – |
| bank-marketing | gpt-6-luna #3 | – | – | – |
| house-prices | gpt-6-luna #1 | Alley, GarageYrBlt, Utilities | – | MoSold (none vs cast + onehot), OverallQual (none vs cast + onehot) |
| house-prices | gpt-6-luna #2 | Alley, GarageYrBlt, MiscFeature, PoolQC, Utilities | – | MoSold (none vs cast + onehot) |
| house-prices | gpt-6-luna #3 | Alley, GarageYrBlt, MiscFeature, Utilities | – | MoSold (none vs cast + onehot) |
| employee-salaries | gpt-6-luna #1 | – | – | division (onehot(50) vs onehot), employee_position_title (onehot(100) vs onehot), underfilled_job_title (onehot(10) vs onehot) |
| employee-salaries | gpt-6-luna #2 | – | – | division (onehot(50) vs onehot), employee_position_title (onehot(100) vs onehot), underfilled_job_title (onehot(10) vs onehot) |
| employee-salaries | gpt-6-luna #3 | – | – | division (onehot(50) vs onehot), employee_position_title (onehot(100) vs onehot), underfilled_job_title (onehot(10) vs onehot) |
| bike-sharing-demand | gpt-6-luna #1 | feel_temp | – | year (none vs cast + onehot) |
| bike-sharing-demand | gpt-6-luna #2 | feel_temp | – | – |
| bike-sharing-demand | gpt-6-luna #3 | feel_temp | – | – |

## Agent repeats

Per dataset and model, over the repeats of the agent: how many runs succeeded, how many of the leaky columns the plans dropped (every run counted on its own; a run without a plan is left out) and the standard deviation of the random forest's score on the main metric from run to run (n/a below two scored runs).

| dataset | model | successful runs | leaky columns caught | score std across runs |
| --- | --- | --- | --- | --- |
| titanic | gpt-6-luna | 3 of 3 (100%) | 6 of 6 (100%) | 0.006 |
| credit-g | gpt-6-luna | 3 of 3 (100%) | – | 0.009 |
| diabetes | gpt-6-luna | 3 of 3 (100%) | – | 0.000 |
| telco-customer-churn | gpt-6-luna | 3 of 3 (100%) | – | 0.003 |
| adult | gpt-6-luna | 3 of 3 (100%) | – | 0.001 |
| bank-marketing | gpt-6-luna | 3 of 3 (100%) | – | 0.000 |
| house-prices | gpt-6-luna | 3 of 3 (100%) | – | 0.002 |
| employee-salaries | gpt-6-luna | 3 of 3 (100%) | 3 of 3 (100%) | 0.000 |
| bike-sharing-demand | gpt-6-luna | 3 of 3 (100%) | – | 0.000 |

## Agent runs

| model | runs | ok | revisions | schema retries | questions | input tokens | output tokens | cost (USD) | plan seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | 27 | 27 (100%) | 0.1 | 0.0 | 0.3 | 10602 | 1574 | 0.0302 | 19.2 |

## Most frequent issue codes

### First-plan errors of agent runs

| code | runs | share of runs |
| --- | --- | --- |
| encode_numeric | 3 | 11% |
| unencoded_column_remains | 1 | 4% |

### Final-plan warnings of agent runs

| code | runs | share of runs |
| --- | --- | --- |
| high_association_kept | 6 | 22% |

### Warnings of reference and naive plans

| dataset | reference | naive |
| --- | --- | --- |
| titanic | – | personal_data_kept, suspected_leakage_kept |
| credit-g | – | – |
| diabetes | – | – |
| telco-customer-churn | train_test_overlap | train_test_overlap |
| adult | – | train_test_overlap |
| bank-marketing | train_test_overlap | train_test_overlap |
| house-prices | high_association_kept | high_association_kept |
| employee-salaries | high_association_kept | high_association_kept, personal_data_kept |
| bike-sharing-demand | train_test_overlap | train_test_overlap |

## Failures

No failures.

## Notes

- Scores come from the baseline module: 5-fold cross-validation (fewer folds when a class is small) on each plan's own train split. The test set is not used. The spread between folds is in results.json; a difference smaller than that spread is not evidence that one plan is better.
- Plans differ in the rows they keep and the columns they produce, so a score compares what a plan hands to the same models, not the models themselves. The logistic regression may stop before it converges on unscaled features; that lowers the score of a plan without scaling and is not reported separately.
- The reference plans are hand-written and were reviewed by a human; see reference/README.md for the decisions behind them.
- The naive plan is rule-based: drop identifier and constant columns, split, median and most-frequent imputation, one-hot encoding of every non-numeric column.
- Cost is the token use the provider reported times the prices in prices.yaml; n/a means the provider reported no usage.
- Cost of gpt-6-luna: cached input and cache writes are priced separately.
