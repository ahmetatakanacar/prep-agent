# Planted-fault datasets

Eight small synthetic datasets with faults planted on purpose, a ground truth for every column, and a scorer that judges a plan against that truth. They sit next to the real-data benchmark (`benchmarks/`, `docs/benchmark.md`) and change nothing in it.

## Held-out: do not tune on this set

**These datasets are held out. Do not use them to tune the planner prompt, the validator rules or the profiler thresholds.** The point of the set is to measure the agent on faults it was not fitted to; if a rule is written because a planted dataset failed, the set no longer measures that and the number it gives is worthless. When a run on this set shows a weakness, write the fix against a new dataset of your own, or change the generator's seeds and recipes only as part of building a *new* held-out set, and say so in the commit.

## What is in a dataset

A realistic base table (4 to 7 numeric and 2 to 3 categorical features, some of them with missing values, and a target that depends on them) plus faults. Feature names are `c01`, `c02`, ... in shuffled order and the target is `y`. Names carry no meaning on purpose: a model cannot recognise a fault from `email` or `customer_id`, and the profiler's name rules (for identifiers and personal data) are not exercised. What is left to decide on is the content of the column.

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

`data/<name>.truth.json` has, for every feature column:

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

The recipes (seed, task, size and faults of each dataset) are in `planted_generate.py`. A test regenerates every dataset and fails if the result differs from the committed files, so the files and the code cannot drift apart.

## Scoring a plan

`planted_scorer.score_plan(plan, truth)` judges the decisions in a plan, not the data it produces: which columns `drop_columns` removes and which `cast_to_category` converts. It gives:

- **Detection rate per fault type**: the share of columns of that type the plan handled, with the count behind it. A `must_drop` column is handled when dropped. A duplicate pair is handled when at least one of the two is dropped. A `numeric_category` column is handled when it is cast and not dropped. The overall detection rate is the same ratio over all measurable faults.
- **Drop precision and recall**. A drop is right when it removes a `must_drop` column or one of a duplicate pair, and wrong when it removes a `must_keep` column. Dropping both columns of a pair is one right drop and one wrong drop. Dropping an `either` column is neutral. Precision is `None` when the plan drops nothing that counts, recall is `None` when the data needs no drop.
- **Wrongly dropped `must_keep` columns** (`over_dropped`) and pairs dropped as a whole (`pairs_dropped_both`).
- **Expected op applied**: how many columns with an `expected_op` got it (`op_rate`, `missing_ops`), and the columns that got an `avoid_op` (`avoided_op_violations`).
- **Unmeasurable columns** and any dropped column the truth does not know (`unknown_dropped`).

`combine` adds the counts of several scores before taking ratios, so a rate over the whole set is a micro average: a dataset with more faults weighs more. With 8 datasets most fault types occur 2 to 4 times, so a single run's per-type rate moves in steps of 25 to 50 percentage points; read it together with its count.

The scorer says nothing about how well the rest of the plan prepares the data (imputation, encoding, scaling). That is what the main benchmark's cross-validated scores are for.
