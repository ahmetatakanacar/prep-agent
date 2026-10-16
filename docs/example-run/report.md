# Preparation report

- **Goal:** Predict which passengers survived
- **Input:** titanic.csv (sha256 b83ab0b5169c)
- **Date:** 2026-10-16 19:14 UTC
- **Model:** gpt-6-luna
- **Token usage:** 11810 input (3771 cached, 8033 cache writes) / 2984 output
- **Target:** survived (classification, model family any)
- **Status:** ok

Text the user wrote (the goal and answers) and the planner's questions and reasons are kept in the language they were written in; everything else in this report is in English.

## Data summary

| | Rows | Columns |
|---|---:|---:|
| Input | 1309 | 14 |
| Prepared (train + test) | 1309 | 33 |
| Train | 1047 | 33 |
| Test | 262 | 33 |

- **Dropped columns:** name, boat, body, cabin, home.dest

## Questions and answers

- **Which model family should the preparation support?** Answer: any. Why it was asked: The model family was not specified, and it affects how numeric features are scaled and categorical features are encoded.
- **Should the cabin column be kept as a feature despite 77% of its values being missing?** Answer: Drop it. Why it was asked: The profile shows that most cabin values are missing. Whether the remaining cabin information is worth retaining cannot be determined from the profile alone.
- **Should the home.dest column be kept as a feature despite 43% of its values being missing?** Answer: Drop it. Why it was asked: The profile shows substantial missingness and many distinct destinations. Whether this information is useful enough to retain is a modeling choice that cannot be settled from the profile alone.

## Plan

| Step | Op | Columns | Parameters | Reason |
|---:|---|---|---|---|
| 0 | drop_columns | name, boat, body, cabin, home.dest |  | These columns are unsuitable for predicting survival before the outcome: name is high-cardinality personal information, boat and body record post-outcome information, and cabin and home.dest are being dropped as requested. |
| 1 | cast_to_category | pclass |  | pclass represents passenger class rather than a measured quantity, so treat its values as categories instead of implying numeric distances between classes. |
| 2 | split |  | test_size=0.2, stratify=true, random_state=42 | Create a held-out test set for evaluating classification performance while preserving the target's class proportions. |
| 3 | impute | age, fare | strategy=median, add_indicator=false | age and fare have missing values; their medians are suitable imputations, particularly given fare's strong skew and many outliers. |
| 4 | impute | embarked | strategy=most_frequent, add_indicator=false | embarked has two missing values, so fill them with the most frequent observed port before encoding. |
| 5 | scale | age, sibsp, parch, fare | method=robust | fare has many outliers, so robust scaling is appropriate for the numeric predictors and also supports linear models. |
| 6 | encode | pclass, sex, ticket, embarked | method=onehot, max_categories=20 | Convert pclass, sex, ticket, and embarked into numeric features for modeling; cap ticket's one-hot expansion by grouping its rarest categories. |

## Validation

Plan validation: no issues.

Result validation: no issues.

## Baseline

Cross-validation on the train set (1047 rows): 5-fold stratified, shuffled, random_state 42. Cells are the mean ± the standard deviation over the folds.

The dummy model ignores the features and always predicts the most frequent class, so its score is the reference: the other models only show the data is usable if they beat it.

| Model | accuracy ↑ | f1_macro ↑ | roc_auc ↑ |
|---|---:|---:|---:|
| dummy_most_frequent | 0.6180 ± 0.0009 | 0.3819 ± 0.0003 | 0.5000 ± 0.0000 |
| logistic_regression | 0.7822 ± 0.0125 | 0.7658 ± 0.0155 | 0.8429 ± 0.0116 |
| random_forest | 0.7927 ± 0.0150 | 0.7790 ± 0.0178 | 0.8377 ± 0.0203 |

## Notes

- Stateful steps (impute, encode, scale) were fit on the train set only; the test set was only transformed with what train taught them.
- The test set was not touched by anything else: no baseline and no model comparison uses it.
- The baselines are computed by cross-validation on the train set. Every fold fits a fresh copy of the stateful steps on its own training part, so nothing leaks from the validation rows.

## Reproduce

`pipeline.py` in this run folder repeats the whole preparation on the raw input file. It needs only pandas and scikit-learn; prep-agent does not have to be installed.

```
python pipeline.py <input.csv> --out-dir <folder>
```

It writes `train.csv` and `test.csv` to the folder (the current directory when `--out-dir` is left out). Run on the same input file (titanic.csv, sha256 b83ab0b5169c) with the pandas and scikit-learn versions named in the header of the script, they are identical to the train and test files in this run folder.
