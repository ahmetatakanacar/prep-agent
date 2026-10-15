# Reference plans

**Status of the plans in this folder: all reviewed by a human.** Every dataset section below starts with its own `Review status` line. The questions that came up in the review and the decisions on them are in `REVIEW.md`.

Each `<dataset>.plan.json` is the plan an experienced data scientist might write for the dataset in `benchmarks/datasets.yaml`, written from the dataset's profile and a look at its columns. They are the benchmark's yardstick for the agent, so a wrong decision here becomes a wrong verdict on the agent. Read the `reason` of each step, then the notes below, and change what you disagree with. The files are plain plan JSON; edit them by hand.

To check an edited plan against the data without scoring it:

```powershell
uv run prep-agent run benchmarks/datasets/titanic.csv --plan benchmarks/reference/titanic.plan.json --dry-run
```

All plans use `model_family: "any"`, a stratified 80/20 split for classification and a plain one for regression, and `random_state` 42, the same as the naive plan.

## Decisions that apply to every plan

- **Scaling under "any".** The planner's system prompt tells the agent to scale the numeric columns (robust when there are many outliers) and to one-hot encode when the model family is unknown, because the model may be linear and scaling costs trees nothing. These plans do the same, and the baseline includes a linear model. (An earlier prompt told the agent to skip scaling under "any", which would have made the agent differ from these plans for a reason that has nothing to do with judgement.)
- **Only one-hot.** No plan uses ordinal encoding; the ordinal encoder orders categories alphabetically, which is an invented order for grades like Ex/Gd/TA.
- **No `dedupe`.** Datasets with exact duplicate rows (telco 22, bank-marketing 274, adult 52, bike-sharing 2) have no identifier, so a repeated row may be a different customer or hour with the same values. See adult for what happens when duplicates are removed anyway.
- **High-missing columns are dropped without asking.** A hand-written plan has no user to ask; the agent would have to ask (see CLAUDE.md, rule 5).
- **Known leakage.** `datasets.yaml` lists `leaky_columns` for titanic (`boat`, `body`) and employee-salaries (`2016_gross_pay_received`). Every reference plan drops them (a test checks it), and the results mark any plan that keeps one. The naive plan keeps them and scores far higher on titanic because of it. The employee-salaries entry is a judgement (see below); it was confirmed in the review.
- **Categories stored as numbers.** `cast_to_category` turns such a column into text so that it can be one-hot encoded. A plan uses it where the column name says the numbers are labels (hour, month, a class code) and the profile hints at it with the `low_cardinality_numeric` warning; the warning alone is not enough, since a count of rooms triggers it too. Only bike-sharing-demand and house-prices (`MSSubClass`) use it so far.
- **What no plan can express.** There is no operation to convert numbers stored as text, parse dates, extract text parts, take logarithms, clip outliers or turn sentinel values into missing. Where that matters it is noted below.

## titanic

**Review status:** reviewed by human

Drops `name`, `ticket`, `cabin`, `boat`, `body` and `home.dest`; imputes `age` (median, with a missing indicator), `fare` (median) and `embarked` (most frequent); one-hot encodes `sex` and `embarked`; robust-scales the numeric columns.

- `boat` and `body` are treated as target leakage: a lifeboat is known only for survivors, a body number only for the dead (confirmed in the review).
- The title inside `name` (Mr, Mrs, Miss, Master) is a strong feature, but no operation can extract it, so the whole column goes.
- `age` gets a missing indicator: ages are missing for 29% of third-class passengers against 6% of second-class ones, and survival differs (28% against 41%). This is a taste decision.

## credit-g

**Review status:** reviewed by human

No drops and no imputation (the table is clean). One-hot encodes the 13 categorical columns and robust-scales the 7 numeric ones.

- `checking_status`, `savings_status` and `employment` are ordered scales; one-hot throws the order away.
- The point of this dataset is restraint: the question is whether the agent leaves a clean table alone.

## diabetes

**Review status:** reviewed by human

Only the split and a robust scaling of the 8 numeric columns.

- Zeros in `plas` (5 rows), `pres` (35), `skin` (227), `insu` (374) and `mass` (11) cannot be real measurements; they are missing values. No operation turns a value into missing, so they stay. Dropping `insu` and `skin` is a defensible alternative; the plan keeps them because tree models can still use the zeros as a marker.

## telco-customer-churn

**Review status:** reviewed by human

Drops `TotalCharges`; stratified split; one-hot encodes the 15 text columns; standard-scales `tenure` and `MonthlyCharges`.

- `TotalCharges` is text with 11 non-numeric entries and cannot be converted, so it is dropped. It correlates 0.9996 with `tenure * MonthlyCharges`, so almost nothing is lost.
- The 22 duplicate rows are kept (see above). `PhoneService` is a yes/no column; the profiler used to flag it `possible_pii` because of "phone" in its name and no longer does, since a name only counts when the values are mostly distinct. `SeniorCitizen` is a 0/1 number and is left as it is.

## adult

**Review status:** reviewed by human

Drops `fnlwgt` (census sampling weight) and `education` (repeats `education-num`); fills the missing `workclass`, `occupation` and `native-country` with the category "Unknown"; one-hot encodes the categorical columns, grouping `native-country` into 10; standard-scales the 5 numeric columns.

- The 52 exact duplicate rows are kept, and the result validator reports them as a `train_test_overlap` warning. An early version removed them with `dedupe` before dropping `fnlwgt`, but then the validator reported `train_test_overlap` as an error, because it counted rows that became identical once `fnlwgt` was gone. It now only counts records that are identical in the raw input, so a `dedupe` placed before the drop would be accepted and would remove exactly the 52 true copies. Deduplicating after the drop would delete different people who share their attributes. Whether to keep the copies is still a judgement call, and the plan did not make it.
- "Unknown" versus the most frequent value for missing categories: the missingness is informative here (workclass and occupation go missing together).
- `capital-gain` and `capital-loss` are mostly 0 with a very long tail; standard scaling is crude for them.

## bank-marketing

**Review status:** reviewed by human

Stratified split; one-hot encodes 8 text columns; robust-scales `age`, `balance`, `campaign`, `pdays` and `previous`.

- Only 11.7% of the clients subscribe. The split is stratified; nothing else is done about the imbalance (the agent does not choose models or weights).
- "unknown" is kept as a category (`job`, `education`, `contact`, `poutcome`). `pdays` is -1 for the 82% never contacted before, a sentinel that stays a number; `poutcome` is "unknown" for the same 82%, so the two say the same thing.
- The 274 duplicate rows are kept (see above).

## house-prices

**Review status:** reviewed by human

Drops `Id`, `GarageYrBlt`, `Utilities`, `Alley`, `PoolQC` and `MiscFeature`; fills the missing values of 12 categorical columns with the category "None" (the feature is absent), `Electrical` with the most frequent value, `MasVnrArea` with the median (0) and `LotFrontage` with the median plus an indicator; one-hot encodes every text column; robust-scales every numeric column.

- Most "missing" values here are not lost data: no alley, no pool, no basement, no fence. Treating them as the category "None" is the key call of this plan. Check the 12 columns listed in the first `impute` step. `MasVnrType` is a special case: 864 of its houses have the text "None" (no veneer). The loader used to read that text as missing (872 missing values, 60%); it now keeps it as a category, so only the 8 real gaps are missing (0.5%) and the first `impute` step fills just those.
- `Alley`, `PoolQC` and `MiscFeature` are dropped because they are 94% to 99.5% empty; `Utilities` because one house differs from the rest. Low-count signals are lost.
- `MSSubClass` is a building-class code stored as a number, so a `cast_to_category` step turns it into text and a separate one-hot `encode` step follows (it is no longer scaled). `MoSold` and `YrSold` are calendar values and are still scaled as numbers; `MoSold` is also a category stored as a number; the review decided not to cast it, because the sale month has weak signal.
- The sale price is skewed (1.9) and has no log transform; large houses are outliers that nothing clips.

## employee-salaries

**Review status:** reviewed by human

Drops `full_name`, `department_name`, `underfilled_job_title`, `date_first_hired` and `2016_gross_pay_received`; imputes `gender` (most frequent) and `2016_overtime_pay` (0); one-hot encodes 3 small columns and, grouped to 50 categories, `division` and `employee_position_title`; robust-scales the 2 numeric columns.

- **The biggest call:** `2016_gross_pay_received` correlates 0.89 with the target. The plan treats it as leakage (it is what the person was paid for the salary being predicted) and drops it. A plan that keeps it will score much higher; judge that as leakage and not as quality, (confirmed in the review).
- `date_first_hired` is dropped as unparseable; `year_first_hired` carries the same information.
- A missing `2016_overtime_pay` is read as no overtime (0): 2917 values are missing and only 6 rows hold an explicit 0. That is an assumption.

## bike-sharing-demand

**Review status:** reviewed by human

Drops `feel_temp` (correlation 0.99 with `temp`); casts `hour`, `month` and `weekday` to categories; makes a plain 80/20 split (regression); one-hot encodes `season`, `weather`, `hour`, `month` and `weekday`; standard-scales `temp`, `humidity` and `windspeed`.

- `month`, `hour` and `weekday` are categories stored as integers. An encode step rejects numeric columns, so before the cast step existed they stayed ordered numbers and the linear model scored R² 0.396. Now they are cast to text and one-hot encoded. One-hot ignores that they are cyclic (hour 23 is next to hour 0); that is a known simplification.
- The 2 duplicate rows are kept: two hours can have identical weather and demand.
