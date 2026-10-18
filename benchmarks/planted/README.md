# Planted-fault datasets

Eight small synthetic datasets, each in an anonymous and a named variant, with faults planted on purpose, a ground truth for every column, and a scorer that judges a plan against that truth. They sit next to the real-data benchmark (`benchmarks/`, `docs/benchmark.md`) and change nothing in it.

## Held-out: do not tune on this set

**These datasets are held out. Do not use them to tune the planner prompt, the validator rules or the profiler thresholds.** The point of the set is to measure the agent on faults it was not fitted to; if a rule is written because a planted dataset failed, the set no longer measures that and the number it gives is worthless. When a run on this set shows a weakness, write the fix against a new dataset of your own, or change the generator's seeds and recipes only as part of building a *new* held-out set, and say so in the commit.

## Two sets

There are two sets of eight datasets built from the same recipes (task, size, faults, domain names) with different seeds. `planted-01` to `planted-08` is the first set; it measured the agent (`docs/benchmark/planted-results/`) before the profile had signals for leakage and copies, and it is the set those signals were designed without looking at. `planted-b01` to `planted-b08` is the second set, made after the signals existed to measure the agent with and without them, and it must not be used to tune them either. A third set, `planted-c01` to `planted-c08`, was made for the closing measurement after the last change to the profile and the prompt; nobody looked at it before that run. A run uses the first set unless `--datasets` names the datasets of another one (`--datasets planted-b01,planted-b02,...`).

## What is in a dataset

A realistic base table (4 to 7 numeric and 2 to 3 categorical features, some of them with missing values, and a target that depends on them) plus faults. Each dataset exists in two variants with the same values, the same faults and the same ground truth; only the column names differ:

- **anonymous** (`planted-01.csv`): features are `c01`, `c02`, ... in shuffled order and the target is `y`. Names carry no meaning on purpose: a model cannot recognise a fault from `email` or `customer_id`, and the profiler's name rules (for identifiers and personal data) are not exercised. What is left to decide on is the content of the column.
- **named** (`planted-01.named.csv`): the same columns with realistic names, in Turkish, English or a mix (`musteri_no`, `email`, `applicant_name`, `toplam_odeme`). The names come from `planted_names.py`, one small domain per dataset (telecom, loans, housing, insurance, retail, staff turnover, used cars, hospital stays). Leaks get names that are plausible and do not announce themselves (`son_donem_risk_skoru`, `collections_status`, `claim_risk_index`), and a duplicate is named after its original (`egitim_saati_v2`). A test checks that no name contains words like "leak" or "target".

The difference between the two variants is what the names add. Because the columns are the same, a result on one variant can be set directly against the other. One limit: the numeric columns come from four fixed shapes (around 100, skewed, whole numbers 18 to 79, 0 to 1000), and the names were ordered to fit them, but a value range can still look odd for a domain.

The fault types below are planted in both variants.

| fault type | what is planted | right decision |
| --- | --- | --- |
| `leak_linear` | a number that is the target times a constant plus a little noise | drop |
| `leak_nonlinear` | an exponential of the (standardised) target plus noise | drop |
| `leak_categorical` | a noisy thresholding of the target into 3 (classification) or 4 (regression) levels | drop |
| `id_sequential` | an increasing integer, with a step of 1 or 3 | drop |
| `id_uuid` | a UUID-like string | drop |
| `pii_email` | e-mail addresses built from Turkish and foreign names | drop |
| `pii_phone` | Turkish mobile numbers in three spellings | drop |
| `pii_name` | full names, mostly Turkish (ş, ğ, ı, ö, ü, ç, İ), some foreign | drop |
| `duplicate_exact` | a copy of another numeric column | drop one of the two |
| `duplicate_scaled` | another numeric column times a constant plus an offset | drop one of the two |
| `numeric_category` | 5 to 10 unordered integer codes whose effect on the target has nothing to do with their size | keep, `cast_to_category` |
| `constant` | one value in every row | drop |
| `sentinel` | an informative numeric column where 5 to 10% of the values are `-999` or `9999` | **not scored** |

`count_decoy` is not a fault. It is an informative integer column with the values 1 to 6 that is a count (rooms, children) and must stay a number. It is there to see whether a plan casts every small integer column to a category.

### Types that cannot be scored

`sentinel` is marked `measurable: false` in the ground truth. The op registry has no op that replaces a value with a missing value, so a plan cannot handle the fault; dropping the column and keeping it are both defensible. The columns are in the data, so a later change to the ops can start scoring them, but no score is computed from them now. The scorer lists them as unmeasurable and leaves them out of every rate.

## Ground truth

`data/<name>.truth.json` and `data/<name>.named.truth.json` have, for every feature column:

- `role`: `must_drop`, `must_keep` or `either`.
- `fault`: the fault type, or nothing for a plain column.
- `expected_op`: an op a good plan applies to the column (`cast_to_category` for `numeric_category`).
- `avoid_op`: an op that would be a mistake (`cast_to_category` on a count decoy).
- `duplicate_of`: for a duplicate, the column it copies. The copy and its original are both `either`: a plan has to drop one of them, and it does not matter which.
- `measurable`: false for the types that cannot be scored.

The truth is written by the same code that plants the fault, so the two cannot disagree.

## Regenerating

```powershell
uv run python benchmarks/planted/planted_generate.py
```

The recipes (seed, task, size and faults of each dataset) are in `planted_generate.py`. A test regenerates every dataset in both variants and fails if the result differs from the committed files, so the files and the code cannot drift apart.

## Scoring a plan

`planted_scorer.score_plan(plan, truth)` judges the decisions in a plan, not the data it produces: which columns `drop_columns` removes and which `cast_to_category` converts. It gives:

- **Detection rate per fault type**: the share of columns of that type the plan handled, with the count behind it. A `must_drop` column is handled when dropped. A duplicate pair is handled when at least one of the two is dropped. A `numeric_category` column is handled when it is cast and not dropped. The overall detection rate is the same ratio over all measurable faults.
- **Drop precision and recall**. A drop is right when it removes a `must_drop` column or one of a duplicate pair, and wrong when it removes a `must_keep` column. Dropping both columns of a pair is one right drop and one wrong drop. Dropping an `either` column is neutral. Precision is `None` when the plan drops nothing that counts, recall is `None` when the data needs no drop.
- **Wrongly dropped `must_keep` columns** (`over_dropped`) and pairs dropped as a whole (`pairs_dropped_both`).
- **Expected op applied**: how many columns with an `expected_op` got it (`op_rate`, `missing_ops`), and the columns that got an `avoid_op` (`avoided_op_violations`).
- **Unmeasurable columns** and any dropped column the truth does not know (`unknown_dropped`).

`combine` adds the counts of several scores before taking ratios, so a rate over the whole set is a micro average: a dataset with more faults weighs more. With 8 datasets most fault types occur 2 to 4 times, so a single run's per-type rate moves in steps of 25 to 50 percentage points; read it together with its count.

The scorer says nothing about how well the rest of the plan prepares the data (imputation, encoding, scaling). That is what the main benchmark's cross-validated scores are for.

## Running the agent on the set

```powershell
uv run python benchmarks/run_planted.py --models gpt-6-luna --dry-run
uv run python benchmarks/run_planted.py --models gpt-6-luna --repeats 3
uv run python benchmarks/run_planted.py --summarize benchmarks/results/<id>/results.json
```

A run is the product of datasets, variants (`anonymous`, `named`), answer modes, models and repeats; `--datasets`, `--variants` and `--modes` narrow it. Like the main benchmark, the runner shows the runs and the most they can cost, refuses a model with no price in `benchmarks/prices.yaml`, and asks before it spends anything (`--yes` skips the question, `--dry-run` stops after the listing). The agent is given the dataset's target and the model family `any`, and the same one-sentence goal for both variants. Results go to `benchmarks/results/planted-<timestamp>/` (git-ignored) as `results.json` (one record per run: the plan, the score, tokens, cost, the questions and answers) and `summary.md`. `summary.md` adds the counts of the runs of a group up before it takes a ratio.

### Answer modes

The agent may ask the user questions. Three modes say how a run answers them without a person:

| mode | what happens |
| --- | --- |
| `recommended` | every question is answered with the option the agent recommends (what the main benchmark does) |
| `not_recommended` | every question is answered with the first option the agent did not recommend, so a recommended "drop this column" becomes "keep it". If a question has no other option, the recommended one is taken and the answer is marked `forced` in `results.json` and counted in the summary |
| `no_questions` | the agent may not ask. The question-round limit of the agent is set to zero for the run, so its prompt says "Do not ask any more questions" and it decides alone. No product code is involved |

`recommended` shows what the agent does when its own advice is followed, `not_recommended` shows whether the plan carries out a decision the user made against the agent, and `no_questions` shows what the agent finds on its own. Only the last one measures whether the agent notices a fault without having asked about it.

### The summary

`summary.md` has one row per model, variant and mode (mode and variant are separate columns):

- detection rate over all faults, with the range over repeats when there are several;
- drop precision and recall, `must_keep` columns dropped, `avoided_op` violations;
- questions, calls and cost per run, and the total cost;
- a table of detection per fault type with the counts behind each rate;
- anonymous against named for the same model and mode.

A run that failed or errored has no score. It counts in the `runs` column (`3 (2)` is three runs, two of them scored), is listed at the end, and is not counted as a miss.

## What the profile shows the agent (observed before any change)

The planner sees only the profile (`profile.json`), never the data. Profiling the anonymous set shows how much of each fault the profile gives away. These are observations on the current profiler, prompt and validator, listed so that the first results can be read against them; nothing was changed because of them (the set is held out).

- **Leak columns carry no warning.** None of the three leak types has a profile warning, and the profile has no relation to the target. The only leak check, `possible_target_leakage`, runs after execution, only warns, and needs a numeric target with an absolute correlation above 0.98, so it can only fire on `leak_linear`. An agent finds a leak by asking about it or by guessing from the values, and in the anonymous variant there is no name to go on.
- **UUID columns are typed `text`, not `id_like`.** A 36-character unique string has a mean length above the text threshold, so no `possible_id` warning is raised. The prompt tells the agent to drop text columns, so a UUID is still dropped, for another reason.
- **Person-name columns are not flagged as personal data when the name says nothing.** The `possible_pii` rule works on the column name for names; only e-mail addresses and phone numbers are recognised by their values. In the anonymous variant a `pii_name` column is high-cardinality categorical with no warning; in the named variant `ad_soyad`, `applicant_name` and the like are flagged by name.
- **Integer codes and counts look alike.** `numeric_category` and `count_decoy` both get `low_cardinality_numeric`. The only difference in the profile is the values (codes are spread over 10 to 125 in steps of 5; counts are 1 to 6), and the prompt says to decide by name and target.
- **A sequential id with a step above 1 is typed `numeric`.** The dense-span rule only catches steps of 1, so `id_sequential` with a step of 3 has no `possible_id` warning in the anonymous variant. In the named variant all four id columns are flagged, by name.

In the named variant the profile flags the id columns, the person names, the e-mail addresses and the phone numbers, and still shows nothing for the leaks and the UUIDs (the `uuid` in a name does not make a 36-character string an identifier). Differences between the two variants therefore show how much the agent relies on what the profile flags.

## Set D: partial leaks and legitimate strong features

Sets A, B and C are the same eight recipes with three sets of seeds. Their leaks give themselves away (an association of 0.95 to 1.00) and the legitimate columns are weak (at most about 0.44), so a threshold on the association separates them by construction. Set D asks a harder question and is independent of the others: new domains, column names, column counts and row counts, mixed classification and regression, and the old fault types in less obvious forms.

What it adds are columns that carry part of the target, at a chosen **level** of association with it, with the label set by what the column **means** and not by the number:

- `leak_partial` (must drop): information that exists only after the outcome, such as a status written afterwards, a cost computed from the result, or a date that is filled in only when the event happened. Levels 0.6, 0.75, 0.85 and 0.95.
- `legit_strong` (must keep): a real property that is known before the outcome and predicts it well, like the odor of a mushroom or the floor area of a building. Levels 0.6, 0.75, 0.85, 0.93 and 0.97, so some are as strong as a leak and above the profile's 0.9 threshold.

At each level there are both kinds, made the same way from the target, so nothing in the numbers tells them apart. Each such column has a `level` and a `reason` in the ground truth. The forms are a number, a category, or (for classification) a gap: a value that exists only for part of the positive rows. `planted_generate_d.py` explains how a column is made to have a level, and a test checks the mechanism on targets made for that purpose (within 0.1 of the level, for every task, form and level).

The old faults come in less obvious forms: identifiers that are not dense (`sparse` gaps, a `prefixed` string), a UUID without hyphens (`hex32`), an e-mail address built from an initial, a phone number stored as a 10-digit integer, names written surname first or as an initial, a unit-converted copy rounded so coarsely that its correlation stays under 0.9999 (`coarse`), and a categorical copy under other labels (`relabel`).

### Dev and test

The twelve recipes are split in two, with different domains: `planted-d01` to `planted-d06` (dev, for looking at results) and `planted-t01` to `planted-t06` (test). Both halves have the same level distribution: three leaks at each of the four levels and nine legitimate strong features.

**The test half is run once, at the end.** Only its recipes and its ground truth are in the repository: its CSV files are git-ignored and are not written unless `planted_generate_d.py --with-test-data` is given, and nothing was run on them or profiled before that run. The runner refuses the test datasets unless the run is given `--final-test`. The dev half's data is committed like the other sets.

### What set D adds to the scoring

- **False alarm rate**: the share of `legit_strong` columns a plan dropped, with the same count by level. Dropping one is also a wrong drop (`must_keep dropped`, precision), so it counts there too.
- **Leaks found by level**: for `leak_partial`, found out of planted at each level (0.60, 0.75, 0.85, 0.95), next to the detection by type.
- `summary.md` has a `legit strong dropped` column in the main table and two tables by level (leaks found, legitimate features dropped). They appear only for runs that have such columns.
- The runner refuses the test recipes unless the run is given `--final-test`, and refuses any run whose data files are missing before it calls a model.

```powershell
uv run python benchmarks/planted/planted_generate_d.py          # dev data and all ground truth
uv run python benchmarks/run_planted.py --models gpt-6-luna --datasets planted-d01,planted-d02,planted-d03,planted-d04,planted-d05,planted-d06
```
