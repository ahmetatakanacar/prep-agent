# Reference plan review

Review sheet for the draft reference plans in this folder, everything except `bike-sharing-demand` (already reviewed). Each dataset has a profile summary, the current plan, and numbered questions. Each question gives the facts, what the plan does now, and a suggestion. Each question ends with the decision that was taken. The plan listings show the plans as they were when the review sheet was written; where a decision changed a plan, the decision names the commit.

How the questions were found: from the profile warnings (`low_cardinality_numeric`, `possible_pii`, `high_missing`), the raw duplicate row count, `leaky_columns` in `datasets.yaml`, and a check of the data for other leakage suspects. The profile numbers come from `profile_dataframe` on the CSV in `benchmarks/datasets/`. Other numbers (such as survival rates by missingness) were computed from the same files.

## titanic

Target `survived` (classification), 1,309 rows, 14 columns, 0 raw duplicate rows (0.0%). Leaky columns in `datasets.yaml`: `boat`, `body`.

### Profile

| column | type | missing | unique | warnings |
|---|---|---|---|---|
| `pclass` | numeric | 0.0% | 3 | low_cardinality_numeric |
| `survived` | binary | 0.0% | 2 |  |
| `name` | categorical | 0.0% | 1,307 | high_cardinality, possible_pii |
| `sex` | binary | 0.0% | 2 |  |
| `age` | numeric | 20.1% | 98 |  |
| `sibsp` | numeric | 0.0% | 7 | high_skew, low_cardinality_numeric |
| `parch` | numeric | 0.0% | 8 | high_skew, many_outliers, low_cardinality_numeric |
| `ticket` | categorical | 0.0% | 929 | high_cardinality |
| `fare` | numeric | 0.1% | 281 | high_skew, many_outliers |
| `cabin` | categorical | 77.5% | 186 | high_missing, high_cardinality |
| `embarked` | categorical | 0.2% | 3 |  |
| `boat` | categorical | 62.9% | 27 | high_missing |
| `body` | numeric | 90.8% | 121 | high_missing |
| `home.dest` | categorical | 43.1% | 369 | high_missing, high_cardinality |

### Current plan

1. **drop_columns** (columns=name, ticket, cabin, boat, body, home.dest). name is personal data and unique per passenger; ticket and home.dest are high-cardinality text with no usable structure; cabin is 78% missing. boat (lifeboat) is assigned only to people who survived and body only to people who died, so both leak the target and would not exist when predicting survival.
2. **split** (test_size=0.2; stratify=True; random_state=42). Stratified so the 38% survivors are represented in the same share in train and test.
3. **impute** (columns=age; strategy=median; add_indicator=True). 20% of the ages are missing; the median is robust to the long right tail. The indicator keeps the fact that age was unknown: ages are missing for 29% of the third-class passengers against 6% of the second-class ones, and survival differs (28% against 41%).
4. **impute** (columns=fare; strategy=median; add_indicator=False). One fare is missing; the median is robust to the very skewed fares.
5. **impute** (columns=embarked; strategy=most_frequent; add_indicator=False). Two embarkation ports are missing; the most frequent port is the safe fill.
6. **encode** (columns=sex, embarked; method=onehot). Text categories with two or three values; one-hot avoids an invented order.
7. **scale** (columns=pclass, age, sibsp, parch, fare; method=robust). Fare, sibsp and parch are skewed with outliers, so the robust scaler is used for all numeric columns. The model family is open and may include linear models; scaling costs trees nothing and puts linear models on comparable footing.

### Questions

Duplicates: 0, so no dedupe question. No `possible_id` column.

**titanic-1. pclass, sibsp, parch: number or category?**

- Facts: Flagged `low_cardinality_numeric`: pclass (3 values: 1, 2, 3), sibsp (7), parch (8). The plan scales all three as numbers.
- Suggestion: Keep numeric. pclass is an ordered class (correlation with survival -0.31, monotone); sibsp and parch are counts of relatives.
- Decision: agreed with the suggestion; the plan is unchanged.

**titanic-2. name (`possible_pii`, 1307 distinct of 1309): keep dropped?**

- Facts: The plan drops it. The title inside it (Mr 757, Miss 260, Mrs 197, Master 61) is a strong feature, but no op can extract it.
- Suggestion: Keep dropped: it is personal data and unique per row, and no op can use the title.
- Decision: agreed with the suggestion; the plan is unchanged.

**titanic-3. cabin (77.5% missing): drop?**

- Facts: Missing mostly means no cabin was recorded: 21% missing in first class, 92% in second, 98% in third. Survival is 65% with a cabin and 30% without, but that gap is mostly class, and a recorded cabin may reflect how the records were collected rather than the passenger. The plan drops it.
- Suggestion: Keep dropped. pclass and fare already carry the class effect. The alternative (fill with "Unknown" and encode with `max_categories`) would build a noisy has-cabin flag out of 186 deck codes.
- Decision: agreed with the suggestion; the plan is unchanged.

**titanic-4. home.dest (43.1% missing, 369 distinct): drop?**

- Facts: Free text such as "New York, NY". Survival is 47% when present and 27% when missing, again largely a class effect. The plan drops it.
- Suggestion: Keep dropped: high cardinality, no usable structure.
- Decision: agreed with the suggestion; the plan is unchanged.

**titanic-5. boat and body: confirm as leakage (`leaky_columns`)**

- Facts: Survival is 98.1% when `boat` is present and 2.8% when it is missing; `body` present means 0% survived (a recovered body). Both are only known after the outcome. The plan drops both. No other column looks like a consequence of the target (`ticket` is shared by 596 rows, which is group information, not leakage).
- Suggestion: Confirm leakage and keep both dropped.
- Decision: agreed with the suggestion; the plan is unchanged.

**titanic-6. age (20.1% missing): add a missing indicator?**

- Facts: Below the `high_missing` threshold (40%), so the profile does not flag it, but the plan makes a taste decision: median fill plus indicator. Survival is 41% when age is known and 28% when missing.
- Suggestion: Keep the indicator: the missingness carries signal and costs one column.
- Decision: agreed with the suggestion; the plan is unchanged.

## credit-g

Target `class` (classification), 1,000 rows, 21 columns, 0 raw duplicate rows (0.0%). Leaky columns in `datasets.yaml`: none.

### Profile

| column | type | missing | unique | warnings |
|---|---|---|---|---|
| `checking_status` | categorical | 0.0% | 4 |  |
| `duration` | numeric | 0.0% | 33 | many_outliers |
| `credit_history` | categorical | 0.0% | 5 |  |
| `purpose` | categorical | 0.0% | 10 |  |
| `credit_amount` | numeric | 0.0% | 921 | many_outliers |
| `savings_status` | categorical | 0.0% | 5 |  |
| `employment` | categorical | 0.0% | 5 |  |
| `installment_commitment` | numeric | 0.0% | 4 | low_cardinality_numeric |
| `personal_status` | categorical | 0.0% | 4 |  |
| `other_parties` | categorical | 0.0% | 3 |  |
| `residence_since` | numeric | 0.0% | 4 | low_cardinality_numeric |
| `property_magnitude` | categorical | 0.0% | 4 |  |
| `age` | numeric | 0.0% | 53 |  |
| `other_payment_plans` | categorical | 0.0% | 3 |  |
| `housing` | categorical | 0.0% | 3 |  |
| `existing_credits` | numeric | 0.0% | 4 | low_cardinality_numeric |
| `job` | categorical | 0.0% | 4 |  |
| `num_dependents` | numeric | 0.0% | 2 | many_outliers, low_cardinality_numeric |
| `own_telephone` | binary | 0.0% | 2 |  |
| `foreign_worker` | binary | 0.0% | 2 |  |
| `class` | binary | 0.0% | 2 |  |

### Current plan

1. **split** (test_size=0.2; stratify=True; random_state=42). Stratified: 30% of the applicants are bad credit risks and the sample is small.
2. **encode** (columns=13 columns; method=onehot). All categorical columns are text with few values and no missing values; one-hot keeps them usable by every model family without an invented order.
3. **scale** (columns=duration, credit_amount, installment_commitment, residence_since, age, existing_credits, num_dependents; method=robust). credit_amount and duration have many outliers, so the robust scaler is used. The model family is open and may include linear models; scaling costs trees nothing and puts linear models on comparable footing.

### Questions

Duplicates: 0. No missing values, no `possible_pii`, no `possible_id`. Leakage: no `leaky_columns`, and every column is an application-time attribute; nothing looks like a consequence of the target.

**credit-g-1. installment_commitment, residence_since, existing_credits, num_dependents: number or category?**

- Facts: Flagged `low_cardinality_numeric`: the first three take the values 1 to 4, num_dependents 1 or 2. The plan scales them as numbers.
- Suggestion: Keep numeric. They are small ordered scales or counts (num_dependents is effectively binary, so the question is moot for it).
- Decision: agreed with the suggestion; the plan is unchanged.

**credit-g-2. checking_status, savings_status, employment: one-hot or ordered?**

- Facts: These are ordered scales (`<0`, `0<=X<200`, `>=200`; `<1`, `1<=X<4`, `>=7`), but there is no op for a custom order and the ordinal encoder sorts alphabetically. The plan one-hot encodes them.
- Suggestion: Keep one-hot: a wrong invented order is worse than losing the order.
- Decision: agreed with the suggestion; the plan is unchanged.

**credit-g-3. personal_status and foreign_worker: keep sensitive attributes?**

- Facts: `personal_status` mixes sex and marital status ("male single", "female div/dep/mar"); `foreign_worker` is nationality-related (963 yes, 37 no). The profile does not flag them, and they are legitimate columns of the dataset.
- Suggestion: Keep. The benchmark measures data preparation, not fairness; note it in the report if you want to be explicit.
- Decision: agreed with the suggestion; the plan is unchanged.

## diabetes

Target `class` (classification), 768 rows, 9 columns, 0 raw duplicate rows (0.0%). Leaky columns in `datasets.yaml`: none.

### Profile

| column | type | missing | unique | warnings |
|---|---|---|---|---|
| `preg` | numeric | 0.0% | 17 | low_cardinality_numeric |
| `plas` | numeric | 0.0% | 136 |  |
| `pres` | numeric | 0.0% | 47 | many_outliers |
| `skin` | numeric | 0.0% | 51 |  |
| `insu` | numeric | 0.0% | 186 | high_skew |
| `mass` | numeric | 0.0% | 248 |  |
| `pedi` | numeric | 0.0% | 517 |  |
| `age` | numeric | 0.0% | 52 |  |
| `class` | binary | 0.0% | 2 |  |

### Current plan

1. **split** (test_size=0.2; stratify=True; random_state=42). Stratified: 35% of the patients are positive and the table has only 768 rows.
2. **scale** (columns=preg, plas, pres, skin, insu, mass, pedi, age; method=robust). All eight measurements are numeric on very different scales and insu is skewed, so the robust scaler is used. The model family is open and may include linear models; scaling costs trees nothing and puts linear models on comparable footing.

### Questions

Duplicates: 0. No missing values, no `possible_pii`, no `possible_id`. Leakage: none; all columns are measurements taken before the diagnosis.

**diabetes-1. preg: number or category?**

- Facts: Flagged `low_cardinality_numeric` (17 values, 0 to 17). The plan scales it as a number.
- Suggestion: Keep numeric: it is a count of pregnancies.
- Decision: agreed with the suggestion; the plan is unchanged.

**diabetes-2. Zeros that stand for missing measurements (plas 5, pres 35, skin 227, insu 374, mass 11)**

- Facts: The profile shows 0% missing everywhere, so it does not flag this; only the minimum of 0 hints at it. A blood pressure or BMI of 0 is impossible. `preg` has 111 zeros, which are real. No op turns a value into missing, so the plan leaves the zeros as they are. An alternative is to drop `insu` (49% zeros) and `skin` (30% zeros).
- Suggestion: Keep as is. Tree models can use the zero as a marker, and the agent has no better tool either. Dropping `insu` and `skin` is the one defensible change, but then the plan is no longer comparable with what the agent can do.
- Decision: agreed with the suggestion; the plan is unchanged.

## telco-customer-churn

Target `Churn` (classification), 7,043 rows, 20 columns, 22 raw duplicate rows (0.3%). Leaky columns in `datasets.yaml`: none.

### Profile

| column | type | missing | unique | warnings |
|---|---|---|---|---|
| `gender` | binary | 0.0% | 2 |  |
| `SeniorCitizen` | binary | 0.0% | 2 |  |
| `Partner` | binary | 0.0% | 2 |  |
| `Dependents` | binary | 0.0% | 2 |  |
| `tenure` | numeric | 0.0% | 73 |  |
| `PhoneService` | binary | 0.0% | 2 |  |
| `MultipleLines` | categorical | 0.0% | 3 |  |
| `InternetService` | categorical | 0.0% | 3 |  |
| `OnlineSecurity` | categorical | 0.0% | 3 |  |
| `OnlineBackup` | categorical | 0.0% | 3 |  |
| `DeviceProtection` | categorical | 0.0% | 3 |  |
| `TechSupport` | categorical | 0.0% | 3 |  |
| `StreamingTV` | categorical | 0.0% | 3 |  |
| `StreamingMovies` | categorical | 0.0% | 3 |  |
| `Contract` | categorical | 0.0% | 3 |  |
| `PaperlessBilling` | binary | 0.0% | 2 |  |
| `PaymentMethod` | categorical | 0.0% | 4 |  |
| `MonthlyCharges` | numeric | 0.0% | 1,585 |  |
| `TotalCharges` | categorical | 0.0% | 6,531 | high_cardinality |
| `Churn` | binary | 0.0% | 2 |  |

### Current plan

1. **drop_columns** (columns=TotalCharges). TotalCharges is stored as text with 11 non-numeric entries and no step can convert it. It correlates 0.9996 with tenure times MonthlyCharges, which are kept, so almost nothing is lost.
2. **split** (test_size=0.2; stratify=True; random_state=42). Stratified: only 26% of the customers churn.
3. **encode** (columns=15 columns; method=onehot). Every remaining text column has two to four values and no missing values; one-hot is the family-neutral encoding. PhoneService is a yes/no column, not personal data.
4. **scale** (columns=tenure, MonthlyCharges; method=standard). Tenure (months) and MonthlyCharges are on different scales and neither has strong outliers. The model family is open and may include linear models; scaling costs trees nothing and puts linear models on comparable footing.

### Questions

No `low_cardinality_numeric`, `possible_pii` or `possible_id` columns. `SeniorCitizen` is a 0/1 number the profile types as binary; the plan neither scales nor encodes it, so it passes through unchanged. Leakage: no `leaky_columns`, and nothing looks like a consequence of churn. `tenure`, `Contract` and `MonthlyCharges` are legitimate predictors.

**telco-customer-churn-1. TotalCharges (text, 6531 distinct): keep dropped?**

- Facts: Stored as text because 11 entries are the string ' '. All 11 are customers with tenure 0 (new customers, none churned). The column correlates 0.9996 with tenure times MonthlyCharges. No op can convert it, so the plan drops it.
- Suggestion: Keep dropped: almost nothing is lost because tenure and MonthlyCharges stay.
- Decision: agreed with the suggestion; the plan is unchanged.

**telco-customer-churn-2. 22 exact duplicate rows (0.3%): dedupe?**

- Facts: There is no customer id, so a repeated row may be a different customer with the same values. The 42 rows involved churn 24 yes to 18 no (57% against 26% overall). The plan keeps them.
- Suggestion: Keep. The share is tiny, and removing rows on a guess is worse than leaving 22 rows. If you dedupe, put `dedupe` before the split.
- Decision: agreed with the suggestion; the plan is unchanged.

## adult

Target `class` (classification), 48,842 rows, 15 columns, 52 raw duplicate rows (0.1%). Leaky columns in `datasets.yaml`: none.

### Profile

| column | type | missing | unique | warnings |
|---|---|---|---|---|
| `age` | numeric | 0.0% | 74 |  |
| `workclass` | categorical | 5.7% | 8 |  |
| `fnlwgt` | numeric | 0.0% | 28,523 |  |
| `education` | categorical | 0.0% | 16 |  |
| `education-num` | numeric | 0.0% | 16 | low_cardinality_numeric |
| `marital-status` | categorical | 0.0% | 7 |  |
| `occupation` | categorical | 5.8% | 14 |  |
| `relationship` | categorical | 0.0% | 6 |  |
| `race` | categorical | 0.0% | 5 |  |
| `sex` | binary | 0.0% | 2 |  |
| `capital-gain` | numeric | 0.0% | 123 | high_skew, many_outliers |
| `capital-loss` | numeric | 0.0% | 99 | high_skew |
| `hours-per-week` | numeric | 0.0% | 96 | many_outliers |
| `native-country` | categorical | 1.8% | 41 |  |
| `class` | binary | 0.0% | 2 |  |

### Current plan

1. **drop_columns** (columns=fnlwgt, education). fnlwgt is the census sampling weight, a property of the survey and not of the person. education repeats education-num one to one, so the numeric version is kept.
2. **split** (test_size=0.2; stratify=True; random_state=42). Stratified: 24% of the people earn more than 50K.
3. **impute** (columns=workclass, occupation, native-country; strategy=constant; fill_value=Unknown; add_indicator=False). These columns are missing for 2% to 6% of the people and the missingness is itself informative (unknown workclass and occupation go together), so it becomes its own category.
4. **encode** (columns=workclass, marital-status, occupation, relationship, race, sex; method=onehot). Text categories with up to 14 values; one-hot is the family-neutral encoding.
5. **encode** (columns=native-country; method=onehot; max_categories=10). 41 countries, 90% of them the United States; rare countries are grouped to avoid dozens of near-empty columns.
6. **scale** (columns=age, education-num, capital-gain, capital-loss, hours-per-week; method=standard). Numeric columns are on very different scales (capital-gain reaches the tens of thousands). The model family is open and may include linear models; scaling costs trees nothing and puts linear models on comparable footing.

### Questions

No `possible_pii` or `possible_id` columns. Leakage: no `leaky_columns`; nothing is derived from income. `relationship` almost determines `sex` (19,715 of 19,716 "Husband" are male, 2,328 of 2,331 "Wife" are female), which is redundancy, not leakage.

**adult-1. education-num: number or category?**

- Facts: Flagged `low_cardinality_numeric` (16 values, 1 to 16). It is a one-to-one numeric code for `education` (checked), and the plan drops `education` and keeps the number.
- Suggestion: Keep numeric: it is an ordered scale of schooling, and one column is cheaper than 16.
- Decision: agreed with the suggestion; the plan is unchanged.

**adult-2. 52 exact duplicate rows (0.1%): dedupe?**

- Facts: The 101 rows involved are 89 "<=50K" and 12 ">50K". The plan keeps them, and the result validator reports a `train_test_overlap` warning. Unlike telco and bank-marketing, this table has `fnlwgt`, a census weight with 28,523 distinct values; a row identical in all 15 columns including `fnlwgt` is very likely a copy. A `dedupe` placed before the `fnlwgt` drop removes exactly those copies and is accepted by the validator.
- Suggestion: Add `dedupe` before the `drop_columns` step. The effect on scores is tiny, but the overlap warning goes away, and `fnlwgt` is the evidence that these are copies, which is not the case in the other datasets.
- Decision: Dedupe. A `dedupe` step (all columns, keep first) now comes before `drop_columns`; its reason says that a row identical in all columns including `fnlwgt` is almost certainly a copy. Plan changed in commit 0e7333f.

**adult-3. workclass (5.7%), occupation (5.8%), native-country (1.8%) missing: fill with "Unknown"?**

- Facts: Below the `high_missing` threshold. workclass and occupation are missing together (2,799 rows of about 2,800 each). Income ">50K" is 9.4% when occupation is missing against 24.8% otherwise, so the missingness carries signal. The plan fills the three with the category "Unknown".
- Suggestion: Keep "Unknown": it keeps the signal and avoids inventing a most frequent occupation.
- Decision: agreed with the suggestion; the plan is unchanged.

**adult-4. fnlwgt and education dropped: confirm?**

- Facts: `fnlwgt` is the survey sampling weight (correlation with the target -0.006), a property of the sampling, not of the person. `education` repeats `education-num` one to one.
- Suggestion: Confirm both drops.
- Decision: agreed with the suggestion; the plan is unchanged.

**adult-5. capital-gain and capital-loss: mostly 0 with a top-coded value**

- Facts: Flagged `high_skew`. 8.3% of rows have a capital gain and 4.7% a loss; 244 rows hold 99999, which looks like a top-code or placeholder, not a real amount. No op can clip, log or flag it; the plan standard-scales both.
- Suggestion: Keep. Robust scaling would not help (the IQR is 0). Note the 99999 in the report.
- Decision: agreed with the suggestion; the plan is unchanged.

**adult-6. sex, race, relationship: keep sensitive attributes?**

- Facts: The profile does not flag them. They are legitimate columns of the dataset, and the plan one-hot encodes all three.
- Suggestion: Keep. The benchmark measures data preparation, not fairness.
- Decision: agreed with the suggestion; the plan is unchanged.

## bank-marketing

Target `SubscribeTermDeposit` (classification), 45,211 rows, 14 columns, 274 raw duplicate rows (0.6%). Leaky columns in `datasets.yaml`: none.

### Profile

| column | type | missing | unique | warnings |
|---|---|---|---|---|
| `age` | numeric | 0.0% | 77 |  |
| `job` | categorical | 0.0% | 12 |  |
| `marital` | categorical | 0.0% | 3 |  |
| `education` | categorical | 0.0% | 4 |  |
| `default` | binary | 0.0% | 2 |  |
| `balance` | numeric | 0.0% | 7,168 | high_skew, many_outliers |
| `housing` | binary | 0.0% | 2 |  |
| `loan` | binary | 0.0% | 2 |  |
| `contact` | categorical | 0.0% | 3 |  |
| `campaign` | numeric | 0.0% | 48 | high_skew, many_outliers |
| `pdays` | numeric | 0.0% | 559 | high_skew, many_outliers |
| `previous` | numeric | 0.0% | 41 | high_skew, many_outliers |
| `poutcome` | categorical | 0.0% | 4 |  |
| `SubscribeTermDeposit` | binary | 0.0% | 2 |  |

### Current plan

1. **split** (test_size=0.2; stratify=True; random_state=42). Stratified: only 11.7% of the clients subscribe, so an unstratified split could leave the test set with few positives.
2. **encode** (columns=job, marital, education, default, housing, loan, contact, poutcome; method=onehot). Text categories with two to twelve values and no missing values. 'unknown' is kept as a category of its own because a client whose job or contact is unknown may behave differently.
3. **scale** (columns=age, balance, campaign, pdays, previous; method=robust). balance, campaign, pdays and previous are heavily skewed with outliers (pdays is -1 for the 82% never contacted), so the robust scaler is used. The model family is open and may include linear models; scaling costs trees nothing and puts linear models on comparable footing.

### Questions

No `low_cardinality_numeric`, `possible_pii` or `possible_id` columns, and no missing values ("unknown" is a category, see below).

**bank-marketing-1. 274 exact duplicate rows (0.6%): dedupe?**

- Facts: There is no client id. The 494 rows involved are 488 "no" and 6 "yes" (1.2% against 11.7% overall), so they are mostly low-information profiles (low balance, unknown contact); that suggests common profiles rather than copies, but it cannot be proven. The plan keeps them.
- Suggestion: Keep: with no identifier, two clients can share every value.
- Decision: agreed with the suggestion; the plan is unchanged.

**bank-marketing-2. pdays = -1 sentinel (36,954 rows, 81.7%)**

- Facts: -1 means never contacted before. It matches `previous` = 0 exactly (checked) and `poutcome` = "unknown" in all but 5 rows. The plan robust-scales `pdays` as a number, so a linear model reads -1 as "one day before". No op turns the sentinel into missing.
- Suggestion: Keep. `previous` and one-hot `poutcome` carry the never-contacted flag, so the linear model can correct for it. Dropping `pdays` is the alternative.
- Decision: agreed with the suggestion; the plan is unchanged.

**bank-marketing-3. "unknown" categories: job 288, education 1,857, contact 13,020 (28.8%), poutcome 36,959**

- Facts: The plan keeps "unknown" as a category of its own. It carries signal: the yes rate is 4.1% when contact is unknown against 14.9% for cellular.
- Suggestion: Keep as a category.
- Decision: agreed with the suggestion; the plan is unchanged.

**bank-marketing-4. Leakage check: no `duration` column, but `campaign` is borderline**

- Facts: The classic leak of this dataset, call `duration`, is not in this 14-column version (checked). `campaign` counts the contacts made to this client during the current campaign, which is only known once the campaign is over (correlation with the target -0.07). `pdays`, `previous` and `poutcome` describe an earlier campaign and are fine.
- Suggestion: Keep `campaign`: weak, and standard in this dataset. Add it to `leaky_columns` only if you want a stricter reading.
- Decision: agreed with the suggestion; the plan is unchanged.

## house-prices

Target `SalePrice` (regression), 1,460 rows, 81 columns, 0 raw duplicate rows (0.0%). Leaky columns in `datasets.yaml`: none.

### Profile

| column | type | missing | unique | warnings |
|---|---|---|---|---|
| `Id` | id_like | 0.0% | 1,460 | possible_id |
| `MSSubClass` | numeric | 0.0% | 15 | many_outliers, low_cardinality_numeric |
| `MSZoning` | categorical | 0.0% | 5 |  |
| `LotFrontage` | numeric | 17.7% | 110 | high_skew, many_outliers |
| `LotArea` | numeric | 0.0% | 1,073 | high_skew |
| `Street` | binary | 0.0% | 2 |  |
| `Alley` | binary | 93.8% | 2 | high_missing |
| `LotShape` | categorical | 0.0% | 4 |  |
| `LandContour` | categorical | 0.0% | 4 |  |
| `Utilities` | binary | 0.0% | 2 |  |
| `LotConfig` | categorical | 0.0% | 5 |  |
| `LandSlope` | categorical | 0.0% | 3 |  |
| `Neighborhood` | categorical | 0.0% | 25 |  |
| `Condition1` | categorical | 0.0% | 9 |  |
| `Condition2` | categorical | 0.0% | 8 |  |
| `BldgType` | categorical | 0.0% | 5 |  |
| `HouseStyle` | categorical | 0.0% | 8 |  |
| `OverallQual` | numeric | 0.0% | 10 | low_cardinality_numeric |
| `OverallCond` | numeric | 0.0% | 9 | many_outliers, low_cardinality_numeric |
| `YearBuilt` | numeric | 0.0% | 112 |  |
| `YearRemodAdd` | numeric | 0.0% | 61 |  |
| `RoofStyle` | categorical | 0.0% | 6 |  |
| `RoofMatl` | categorical | 0.0% | 8 |  |
| `Exterior1st` | categorical | 0.0% | 15 |  |
| `Exterior2nd` | categorical | 0.0% | 16 |  |
| `MasVnrType` | categorical | 0.5% | 4 |  |
| `MasVnrArea` | numeric | 0.5% | 327 | high_skew, many_outliers |
| `ExterQual` | categorical | 0.0% | 4 |  |
| `ExterCond` | categorical | 0.0% | 5 |  |
| `Foundation` | categorical | 0.0% | 6 |  |
| `BsmtQual` | categorical | 2.5% | 4 |  |
| `BsmtCond` | categorical | 2.5% | 4 |  |
| `BsmtExposure` | categorical | 2.6% | 4 |  |
| `BsmtFinType1` | categorical | 2.5% | 6 |  |
| `BsmtFinSF1` | numeric | 0.0% | 637 |  |
| `BsmtFinType2` | categorical | 2.6% | 6 |  |
| `BsmtFinSF2` | numeric | 0.0% | 144 | high_skew, many_outliers |
| `BsmtUnfSF` | numeric | 0.0% | 780 |  |
| `TotalBsmtSF` | numeric | 0.0% | 721 |  |
| `Heating` | categorical | 0.0% | 6 |  |
| `HeatingQC` | categorical | 0.0% | 5 |  |
| `CentralAir` | binary | 0.0% | 2 |  |
| `Electrical` | categorical | 0.1% | 5 |  |
| `1stFlrSF` | numeric | 0.0% | 753 |  |
| `2ndFlrSF` | numeric | 0.0% | 417 |  |
| `LowQualFinSF` | numeric | 0.0% | 24 | high_skew, low_cardinality_numeric |
| `GrLivArea` | numeric | 0.0% | 861 |  |
| `BsmtFullBath` | numeric | 0.0% | 4 | low_cardinality_numeric |
| `BsmtHalfBath` | numeric | 0.0% | 3 | high_skew, many_outliers, low_cardinality_numeric |
| `FullBath` | numeric | 0.0% | 4 | low_cardinality_numeric |
| `HalfBath` | numeric | 0.0% | 3 | low_cardinality_numeric |
| `BedroomAbvGr` | numeric | 0.0% | 8 | low_cardinality_numeric |
| `KitchenAbvGr` | numeric | 0.0% | 4 | high_skew, low_cardinality_numeric |
| `KitchenQual` | categorical | 0.0% | 4 |  |
| `TotRmsAbvGrd` | numeric | 0.0% | 12 | low_cardinality_numeric |
| `Functional` | categorical | 0.0% | 7 |  |
| `Fireplaces` | numeric | 0.0% | 4 | low_cardinality_numeric |
| `FireplaceQu` | categorical | 47.3% | 5 | high_missing |
| `GarageType` | categorical | 5.5% | 6 |  |
| `GarageYrBlt` | numeric | 5.5% | 97 |  |
| `GarageFinish` | categorical | 5.5% | 3 |  |
| `GarageCars` | numeric | 0.0% | 5 | low_cardinality_numeric |
| `GarageArea` | numeric | 0.0% | 441 |  |
| `GarageQual` | categorical | 5.5% | 5 |  |
| `GarageCond` | categorical | 5.5% | 5 |  |
| `PavedDrive` | categorical | 0.0% | 3 |  |
| `WoodDeckSF` | numeric | 0.0% | 274 |  |
| `OpenPorchSF` | numeric | 0.0% | 202 | high_skew, many_outliers |
| `EnclosedPorch` | numeric | 0.0% | 120 | high_skew, many_outliers |
| `3SsnPorch` | numeric | 0.0% | 20 | high_skew, low_cardinality_numeric |
| `ScreenPorch` | numeric | 0.0% | 76 | high_skew, many_outliers |
| `PoolArea` | numeric | 0.0% | 8 | high_skew, low_cardinality_numeric |
| `PoolQC` | categorical | 99.5% | 3 | high_missing |
| `Fence` | categorical | 80.8% | 4 | high_missing |
| `MiscFeature` | categorical | 96.3% | 4 | high_missing |
| `MiscVal` | numeric | 0.0% | 21 | high_skew, low_cardinality_numeric |
| `MoSold` | numeric | 0.0% | 12 | low_cardinality_numeric |
| `YrSold` | numeric | 0.0% | 5 | low_cardinality_numeric |
| `SaleType` | categorical | 0.0% | 9 |  |
| `SaleCondition` | categorical | 0.0% | 6 |  |
| `SalePrice` | numeric | 0.0% | 663 |  |

### Current plan

1. **drop_columns** (columns=Id, GarageYrBlt, Utilities, Alley, PoolQC, MiscFeature). Id is a row counter. GarageYrBlt tracks YearBuilt (correlation 0.83) and is missing exactly when GarageType is, i.e. when there is no garage, which the other garage columns already say. Utilities has one value except for a single house. Alley, PoolQC and MiscFeature are 94% to 99.5% 'none', too few rows to learn from.
2. **cast_to_category** (columns=MSSubClass). MSSubClass is a building-class code stored as a number (20 is a one-story house built 1946 or later, 60 a two-story one), not an amount: class 120 is not six times class 20. Casting it to text lets it be one-hot encoded, so that a linear model does not read the code as a quantity.
3. **split** (test_size=0.2; stratify=False; random_state=42). Regression: no stratification. 20% of 1460 houses is 292 test rows.
4. **impute** (columns=12 columns; strategy=constant; fill_value=None; add_indicator=False). In this data a missing value means the house has no such feature (no basement, no garage, no fireplace, no fence, no veneer), not that a value was lost, so it becomes the category 'None'. MasVnrType already has 'None' as a category (864 houses without veneer); only its 8 real gaps are filled here, and 'None' matches the area of 0 that MasVnrArea gets for them.
5. **impute** (columns=Electrical; strategy=most_frequent; add_indicator=False). One house has no electrical value; this is a lost value, so the most frequent type is used.
6. **impute** (columns=MasVnrArea; strategy=median; add_indicator=False). Eight houses have no area. For 59% of the houses the area is 0, so the median is 0, the same as 'no veneer'.
7. **impute** (columns=LotFrontage; strategy=median; add_indicator=True). 18% of the lot frontages are missing and the column is skewed with outliers, so the median is used. The indicator keeps the fact that it was unknown.
8. **encode** (columns=39 columns; method=onehot). All remaining text columns are encoded with one-hot. Quality grades (Ex, Gd, TA ...) look ordered, but the ordinal encoder would order them alphabetically, which is wrong, so one-hot is the only encoding that does not invent an order.
9. **encode** (columns=MSSubClass; method=onehot). MSSubClass became text in the cast step and has 15 codes and no missing values; one-hot gives each code its own column and invents no order between the codes.
10. **scale** (columns=34 columns; method=robust). Areas and counts are skewed with many outliers, so the robust scaler is used. The year columns are calendar values and are scaled as numbers. The model family is open and may include linear models; scaling costs trees nothing and puts linear models on comparable footing.

### Questions

Duplicates: 0. No `possible_pii` column.

**house-prices-1. MoSold and YrSold: number or category?**

- Facts: Both flagged `low_cardinality_numeric` (MoSold 1 to 12, YrSold 2006 to 2010). The plan scales both as numbers. README notes that nobody has decided whether MoSold should be cast like `MSSubClass` is.
- Suggestion: Keep YrSold numeric (a year). For MoSold, as a number it is read as a trend, which December-to-January breaks; casting it would model seasonality. Either is defensible; I would not cast it, because the sale month has weak signal (correlation 0.05).
- Decision: agreed with the suggestion; the plan is unchanged.

**house-prices-2. OverallQual, OverallCond and the other flagged small-integer columns: number or category?**

- Facts: `OverallQual` (1 to 10) and `OverallCond` (1 to 9) are ordered ratings; `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`, `Fireplaces`, `GarageCars` are counts; `LowQualFinSF`, `3SsnPorch`, `PoolArea`, `MiscVal` are areas or dollars, mostly 0, flagged only for having few distinct values. The plan scales all as numbers.
- Suggestion: Keep numeric. OverallQual has correlation 0.79 with the price and is monotone; casting a rating loses its order.
- Decision: agreed with the suggestion; the plan is unchanged.

**house-prices-3. Columns dropped: Id, GarageYrBlt, Utilities, Alley, PoolQC, MiscFeature**

- Facts: Id is a counter (`possible_id`). Utilities has one value except for a single house. GarageYrBlt (correlation 0.83 with YearBuilt) is missing exactly when there is no garage (81 houses, all with GarageArea 0). Alley (93.8%), MiscFeature (96.3%) and PoolQC (99.5%) are `high_missing`; there the missing value means no alley, no misc feature, no pool (PoolQC is missing for all 1,453 houses with PoolArea 0). Only 91, 54 and 7 houses are left with a value.
- Suggestion: Keep the drops: too few rows to learn from, and PoolArea and MiscVal stay. If you want consistency with the "missing means absent" rule, `Alley` is the one to turn into "None" instead (91 houses).
- Decision: agreed with the suggestion; the plan is unchanged.

**house-prices-4. Missing means absent: Fence (80.8%), FireplaceQu (47.3%), Garage* (5.5%), Bsmt* (2.5%), MasVnrType (0.5%) filled with "None"**

- Facts: Checked against the numeric columns: FireplaceQu is missing for exactly the 690 houses with Fireplaces 0; the 81 missing GarageType are the 81 with GarageArea 0; the 37 missing BsmtQual are the 37 with TotalBsmtSF 0 (BsmtExposure and BsmtFinType2 have one extra real gap each, filled the same way). Fence has no numeric partner to check against. MasVnrType already has "None" as a text category (864 houses); only its 8 real gaps are filled.
- Suggestion: Confirm. This is the key call of the plan and the data backs it up for everything except Fence, where it is an assumption.
- Decision: agreed with the suggestion; the plan is unchanged.

**house-prices-5. LotFrontage (17.7% missing): median plus indicator?**

- Facts: Not flagged as high missing. Unlike the columns above, a lot always has a frontage, so this is a lost value, not an absent feature. The plan fills the median and adds an indicator.
- Suggestion: Keep.
- Decision: agreed with the suggestion; the plan is unchanged.

**house-prices-6. SaleType and SaleCondition, and the exact sum GrLivArea: leakage or redundancy?**

- Facts: `SaleCondition` "Partial" (median price 244,600 against 160,000 for "Normal") means a home not finished when last assessed, i.e. new construction; it is known at sale time, but describes the sale itself. Separately, GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF exactly (checked), an exact linear dependency; the baseline linear model is Ridge, which copes with it.
- Suggestion: Keep both: no column is derived from the sale price. Only drop SaleType/SaleCondition if the model is meant to value houses before a sale exists.
- Decision: agreed with the suggestion; the plan is unchanged.

## employee-salaries

Target `current_annual_salary` (regression), 9,228 rows, 13 columns, 0 raw duplicate rows (0.0%). Leaky columns in `datasets.yaml`: `2016_gross_pay_received`.

### Profile

| column | type | missing | unique | warnings |
|---|---|---|---|---|
| `full_name` | categorical | 0.0% | 9,222 | high_cardinality, possible_pii |
| `gender` | binary | 0.2% | 2 |  |
| `current_annual_salary` | numeric | 0.0% | 3,403 |  |
| `2016_gross_pay_received` | numeric | 1.1% | 8,977 |  |
| `2016_overtime_pay` | numeric | 31.6% | 6,176 | high_skew, many_outliers |
| `department` | categorical | 0.0% | 37 |  |
| `department_name` | categorical | 0.0% | 37 |  |
| `division` | categorical | 0.0% | 694 | high_cardinality |
| `assignment_category` | binary | 0.0% | 2 |  |
| `employee_position_title` | categorical | 0.0% | 385 | high_cardinality |
| `underfilled_job_title` | categorical | 88.2% | 84 | high_missing, high_cardinality |
| `date_first_hired` | datetime | 0.0% | 2,264 |  |
| `year_first_hired` | numeric | 0.0% | 51 |  |

### Current plan

1. **drop_columns** (columns=full_name, department_name, underfilled_job_title, date_first_hired, 2016_gross_pay_received). full_name is personal data and unique per employee. department_name repeats department one to one. underfilled_job_title is 88% missing. date_first_hired is a date stored as text that no step can parse, and year_first_hired holds the same information. 2016_gross_pay_received is what the employee was paid for the salary being predicted (correlation 0.89), so it would leak the target.
2. **split** (test_size=0.2; stratify=False; random_state=42). The target is continuous, so there are no classes to stratify on; a seeded random 80/20 split keeps the run reproducible.
3. **impute** (columns=gender; strategy=most_frequent; add_indicator=False). 17 genders are missing; the most frequent value is a safe fill for so few rows.
4. **impute** (columns=2016_overtime_pay; strategy=constant; fill_value=0; add_indicator=False). 32% are missing while only 6 rows hold an explicit 0, so a missing overtime pay evidently means no overtime and the fill is 0.
5. **encode** (columns=gender, department, assignment_category; method=onehot). Text categories with two to 37 values; one-hot is the family-neutral encoding.
6. **encode** (columns=division, employee_position_title; method=onehot; max_categories=50). 694 divisions and 385 position titles; the position is the strongest signal for pay, so the 50 most frequent are kept and the rest are grouped.
7. **scale** (columns=2016_overtime_pay, year_first_hired; method=robust). Overtime pay is skewed with many outliers. The model family is open and may include linear models; scaling costs trees nothing and puts linear models on comparable footing.

### Questions

No `low_cardinality_numeric` or `possible_id` columns. Duplicates: 0.

**employee-salaries-1. full_name (`possible_pii`, 9222 distinct of 9228): keep dropped?**

- Facts: Personal data and unique per row. The plan drops it. `gender` (17 missing) is a personal attribute the profile does not flag; the plan keeps it.
- Suggestion: Keep `full_name` dropped. Keeping `gender` is your call; it is a legitimate column of the dataset.
- Decision: agreed with the suggestion; the plan is unchanged. `gender` stays in the plan.

**employee-salaries-2. underfilled_job_title (88.2% missing): drop?**

- Facts: Missing means the employee holds the title they were hired for; a value means they are temporarily in a lower title. This is informative: median salary is 53,788 when it is set against 73,296 when missing. The plan drops the column as 88% empty.
- Suggestion: Consider filling with "None" and encoding it grouped (`max_categories` about 10), consistent with the "missing means absent" rule of house-prices. Dropping is the simpler choice; I lean to the fill.
- Decision: Do not drop. `underfilled_job_title` is filled with "None" (constant) and one-hot encoded with `max_categories` 10; the reason explains that a missing value means the employee holds the title they were hired for. Plan changed in commit 4fcee87.

**employee-salaries-3. 2016_overtime_pay (31.6% missing): is missing really 0?**

- Facts: The plan fills 0 because only 6 rows hold an explicit 0. But missingness is concentrated by department (100% missing for ZAH, 7% for DLC), and the missing rows have a higher median salary (79,382 against 66,065). That fits departments that do not record overtime, not employees who had none.
- Suggestion: Keep the fill of 0 but set `add_indicator` to true, so that the department-level pattern is not erased.
- Decision: Keep the fill of 0 and set `add_indicator` to true; the reason now says the missing values are concentrated in a few departments. Plan changed in commit 4fcee87.

**employee-salaries-4. 2016_gross_pay_received: confirm as leakage (`leaky_columns`), and what about 2016_overtime_pay?**

- Facts: Gross pay correlates 0.89 with the target; the ratio gross/salary has a median of 1.04. It is what was paid for the salary being predicted. The plan drops it. `2016_overtime_pay` is a component of that gross pay and is also earned in the same year (gross minus overtime correlates 0.96 with salary); its own correlation with the target is only 0.28. The plan keeps it.
- Suggestion: Confirm gross pay as leakage. Keep overtime: it is a weaker, partial signal, and dropping it too would be a stricter reading. Your call.
- Decision: agreed with the suggestion; the plan is unchanged.

**employee-salaries-5. division (694 values) and employee_position_title (385): is grouping to the top 50 enough?**

- Facts: Flagged `high_cardinality`. The top 50 cover 47% of the rows for division and 74% for position title; the rest becomes one "other" group. The README calls position the strongest signal for pay.
- Suggestion: Raise `max_categories` for `employee_position_title` to 100 (covers 88%); keep 50 for `division`.
- Decision: `employee_position_title` gets `max_categories` 100 in its own encode step; `division` keeps 50 in another. Plan changed in commit 4fcee87.

**employee-salaries-6. department_name and date_first_hired dropped: confirm?**

- Facts: `department_name` is one to one with `department` (37 each, checked). `date_first_hired` is a date stored as text with no usable op; `year_first_hired` equals its year in 100% of the rows.
- Suggestion: Confirm both drops.
- Decision: agreed with the suggestion; the plan is unchanged.

## Tool gaps found in review

Things a good plan needs but the op registry cannot express. Each one is a limit of the tools, not of the plans, and each is a candidate for a new op (or, if the agent can never do it, a known limit to state in the report).

1. **Turn special values into missing.** A sentinel or placeholder that is not a real measurement stays a number, and imputation never sees it.
   - diabetes zeros in `plas`, `pres`, `skin`, `insu`, `mass` (diabetes-2)
   - bank-marketing `pdays` = -1 for clients never contacted (bank-marketing-2)
   - adult `capital-gain` = 99999, a top-code (adult-5)
2. **Ordinal encoding in a given order.** The ordinal encoder sorts alphabetically, so ordered scales are one-hot encoded and the order is lost.
   - credit-g `checking_status`, `savings_status`, `employment` (credit-g-2)
   - house-prices quality grades (`ExterQual`, `KitchenQual`, `BsmtQual` and others, Ex/Gd/TA/Fa/Po). No question was written for these; they appear in the one-hot `encode` step of the plan and in the README.
3. **Convert text to numbers.** A numeric column stored as text can only be dropped.
   - telco-customer-churn `TotalCharges` (telco-customer-churn-1)
4. **Extract features from text.** Structure inside a text column cannot be used, so the whole column is dropped.
   - titanic title in `name`: Mr, Mrs, Miss, Master (titanic-2)

Also seen in the questions, not on your list:

- **Parse dates.** employee-salaries `date_first_hired` is dropped because no op parses it; `year_first_hired` happens to hold the same information (employee-salaries-6).
- **Clip or transform skewed values.** adult `capital-gain` and `capital-loss` are only standard-scaled (adult-5).
