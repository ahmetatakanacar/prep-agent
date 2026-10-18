# Evaluation

How the agent was evaluated, in the order it happened, what each step showed, and what the evidence does not cover. The tables of the OpenML benchmark are in [benchmark.md](benchmark.md) and the planted datasets are described in [benchmarks/planted/README.md](../benchmarks/planted/README.md). Every number below is copied from a result file kept in the repository; the folder is named at each table and listed at the end. All runs are gpt-6-luna with 3 repeats unless a row says otherwise.

## In short

- Where a leak gives itself away (an association of 0.95 or more with the target), the agent finds almost all of them: 0.97 (anonymous) and 1.00 (named) of the planted faults on set C. This was the original test, and it is easy by construction.
- Where a leak is partial and a legitimate feature can be just as strong (set D), the score does not tell them apart and neither does anything else in an anonymous table. The agent drops what it cannot tell apart. On the test half, a user who accepts every recommendation loses 24 of 27 legitimate strong features in the anonymous variant and 13 of 27 in the named one, and finds every leak from the 0.75 level up. A user who answers with knowledge of the data gets a plan with 96% of the decisions right in both variants.
- Leaks that score under 0.6 are not marked and are found only by their names (3 of 9 in the named variant, 0 of 9 in the anonymous one).
- On the nine OpenML datasets, 27 of 27 runs end with a valid plan and the last run kept none of the 9 leaky columns. Its scores are within noise of the reference plans except on two datasets where the agent scored higher, which says little about preparation quality (see below).
- The evidence is one model, three repeats per dataset, synthetic data for the part about leaks, and a small number of columns. It shows how the agent behaves on these data, not how often it is right on yours.

## 1. The first benchmark and its weaknesses

The first evaluation was the OpenML benchmark (nine datasets, [benchmark.md](benchmark.md)). Its result was good (27 of 27 valid plans, all 9 leaky columns dropped), and it could not show three things:

- Only two datasets have known leaky columns, three columns in all. A rate of finding leaks cannot be measured on three columns.
- Nothing in it counts a legitimate column wrongly dropped, so a tool that drops too much would have looked just as good.
- The agent's questions were answered with its own recommendation. The benchmark measured that it asks, not what it does when a person answers differently.

The other limits are in benchmark.md: one model, scores on the train split, a reference plan drafted by an AI and reviewed by one person, most differences inside noise, and fixes that were made on the same nine datasets.

So the next step was a set of datasets with faults planted on purpose and a ground truth for every column.

## 2. Planted sets A, B and C

Each set has eight small synthetic datasets, each in an anonymous variant (columns `c01`, `c02`, ..., so only the content of a column can be used) and a named variant (same values, realistic names), with the faults and their right answers in a ground truth file. A scorer compares a plan with the truth. The answer modes say how questions are answered without a person: `recommended` (the agent's own advice is followed), `not_recommended` (the first other option), `no_questions` (the agent may not ask). Details are in the planted README.

| set | code | anonymous / named detection, `recommended` | folder |
| --- | --- | --- | --- |
| A | before the profile signals | 0.58 / 0.72 | `planted-results/20261018-030643-gpt-6-luna-3x` |
| B | before the profile signals | 0.57 / 0.72 | `planted-results/20261018-081346-set-b-old-code-gpt-6-luna-3x` |
| B | with the profile signals (`d584967`) | 0.82 / 0.97 | `planted-results/20261018-081347-set-b-new-code-gpt-6-luna-3x` |
| C | with the signals and the prompt rule on missing marks (`25418da`) | 0.90 / 1.00 | `planted-results/20261018-093035-set-c-final-code-gpt-6-luna-3x` |
| C | final code (`1402ac1`) | 0.97 / 1.00 | `planted-results/20261018-200113-set-c-closing-final-code-gpt-6-luna-3x` |

On set A, before the profile had any signal about the target, the agent found none of the planted leaks in the anonymous variant (`leak_linear`, `leak_nonlinear`, `leak_categorical` all 0%) and a fifth to a half in the named variant (22%, 44% and 50% of the three types), from names alone. The signals (target association, duplicates, hash identifiers) were written to answer that.

**These three sets are not independent evidence.**

- They are the same eight recipes (task, size, faults, domains) with three sets of seeds. A change that helps on one helps on the others.
- Set A was run and its profile read before the signals were written, so it motivated them. Set B measured them, and a prompt rule (a missing-value mark does not make a column safe) was added after B and the OpenML run with the signals. Both sets steered the design of what they measured. Set C was made after the last of those changes and was not looked at before its first run, but it is the same recipes.
- They are easy by construction: every leak has an association of 0.95 to 1.00 and the legitimate columns stay under about 0.44, so a threshold on the score separates them. High detection on A to C shows that the agent uses the signals. It does not show that it can tell a leak from a legitimate feature that is as strong.

## 3. The independent review

The owner had the evaluation reviewed independently. The review report is not part of this repository; what is recorded here is the summary of its findings as the owner gave it, and what was done about each.

| finding | what followed |
| --- | --- |
| The planted set is too easy: leaks at 0.95 to 1.00, legitimate columns at 0.44 at most. | Set D: partial leaks at four levels and strong legitimate features up to 0.97 (section 4). |
| A, B and C are the same recipes, not independent. | Set D has its own recipes, domains and names, split into a dev half and a test half that was held out (section 4). |
| False alarms are not measured at all. | A false alarm rate (legitimate strong columns dropped) by level, and drop precision (`7a79ccc`). |
| A plan that keeps a column the profile marked passes without a word. | Validator warnings `suspected_leakage_kept` and `high_association_kept` (`b421f57`); later, a column the user was asked about and chose to keep is reported as the user's decision (`1402ac1`). |
| With an auto-picked target, the signals do not reach the planner. | After the target is known the profile is made again around it, and the planner is called once more with it (`08f4278`). |
| `underfilled_job_title` in employee-salaries is now dropped in 3 of 3 runs, but the reference plan keeps it on purpose. | The prompt says that a high share of missing values is not a reason to drop a column (`75c8b65`). It was kept in 3 of 3 runs in the closing run. |
| Legitimate strong features (the odor of a mushroom at 0.96, the floor area of a house at 0.92) can pass the 0.9 threshold. | Two tiers of marks, 0.6 and 0.9 (`83e6395`), the rule that a mark is not a verdict (`e37e348`), and set D to measure the behavior. |

Other changes made in this phase: an `oracle` answer mode and question metrics in the planted evaluation (`ff8a14f`), a rule for rounded copies (`91fcdbd`), a benchmark noise rule that compares plans fold by fold and counts a run that kept a leak as a loss (`3017aeb`, described in benchmark.md), an upload limit applied while the body is received and a timeout on every LLM request (`eb354f7`).

## 4. Set D: partial leaks and strong legitimate features

Set D asks what the earlier sets could not: what does the agent do with a column that is strongly related to the target, when the numbers cannot say whether it is a consequence of the target or a real property known before it. It has twelve recipes in new domains (the test half, for example, has phone resale, ride cancellation, solar yield and library returns), classification and regression, and the old fault types in less obvious forms.

- `leak_partial` (must drop): information that exists only after the outcome, at association levels 0.6, 0.75, 0.85 and 0.95.
- `legit_strong` (must keep): a real property known before the outcome, at levels 0.6, 0.75, 0.85, 0.93 and 0.97. Some are above the profile's 0.9 mark.
- The label comes from what the column means, not from its score. At each level both kinds exist and are made the same way from the target.

**Dev and test.** The recipes are split into `planted-d01` to `planted-d06` (dev, looked at while correcting) and `planted-t01` to `planted-t06` (test). Each half has 3 leaks at each of the four levels and 9 legitimate strong features. Of the test half only the recipes and ground truth are in the repository; its data was generated for the one run and was not profiled or run before. The runner refuses it without `--final-test`. For both halves the expectations were written and committed before the run: `docs/eval-d-expectations.md` (`1c4f3a4`) and `docs/eval-d-test-expectations.md` (`937f2e6`).

**Oracle mode.** `oracle` answers every question from the ground truth, like a user who knows the data. It separates two things the other modes mix: whether the agent asks about the right columns with options a user can answer (plan accuracy with an informed user), and what it does when nobody answers well. Plan accuracy is the share of the decisions that matter (needed drops and columns to keep) that the plan got right.

### What happened on the dev half, and then on the test half

`recommended` mode, anonymous / named. False alarms are legitimate strong features dropped, out of 27 (9 columns × 3 repeats).

| stage | code | detection | drop precision | false alarms | oracle plan accuracy |
| --- | --- | --- | --- | --- | --- |
| dev, before any correction | `25418da` | 0.61 / 0.86 | 0.79 / 0.83 | 15 / 15 | 0.89 / 0.96 |
| dev, first round | `eb354f7` | 0.66 / 0.82 | 0.80 / 0.92 | 15 / 6 | 0.97 / 0.99 |
| dev, second round | `1402ac1` | 0.87 / 0.96 | 0.77 / 0.91 | 24 / 8 | 0.98 / 0.99 |
| **test, once** | `1402ac1` | **0.87 / 0.94** | **0.78 / 0.87** | **24 / 13** | **0.96 / 0.96** |

Folders: `planted-results/20261018-111329-set-d-dev-final-code-gpt-6-luna-3x` (first row; its oracle value is from `...121249-set-d-dev-old-code-oracle-...`), `...123611-set-d-dev-round1-...`, `...185241-set-d-dev-round2-...`, `...200114-set-d-test-final-code-...`.

Leaks of partial strength found, `recommended`, by level (anonymous; named in the same order):

| stage | 0.60 | 0.75 | 0.85 | 0.95 |
| --- | --- | --- | --- | --- |
| dev, before: anonymous / named | 3/9 / 9/9 | 0/9 / 5/9 | 0/9 / 6/9 | 9/9 / 9/9 |
| dev, first round | 0/9 / 7/9 | 0/9 / 1/9 | 0/9 / 2/9 | 9/9 / 9/9 |
| dev, second round | 3/9 / 5/9 | 9/9 / 9/9 | 9/9 / 9/9 | 9/9 / 9/9 |
| test | 0/9 / 3/9 | 9/9 / 9/9 | 9/9 / 9/9 | 9/9 / 9/9 |

Before the corrections the agent found none of the leaks at 0.75 and 0.85 in the anonymous dev variant and 5 to 6 of 9 in the named one; after the second round it found all of them. It paid for that with false alarms, which is the next section.

## 5. The principle "drop when unclear", and its price

The first round told the planner that a mark is not a verdict and that a marked column is decided by what it means. The second round added the rule for the case where the meaning cannot be told, which is every marked column of an anonymous table:

> A marked column (`suspected_target_leakage` or `high_target_association`) whose meaning the planner cannot tell is dropped by default, with or without questions: it is the recommended option when the planner asks, and what the planner does when it may not ask. A column whose name plainly says it is known before the outcome is still kept, with the reason written down.

(From `docs/eval-d-round2.md`.)

The reason, taken on a principle and not from the dev numbers: keeping a column that leaks the target is a silent error (the scores come out better than the model is, and nobody sees it), and it is the error this tool exists to prevent. Dropping a real feature is a visible loss that can be undone. The rule is a judgement written in the prompt template. The measurements below show its cost; they do not show that it is the right trade.

**The transition condition was violated.** Before the second round, three conditions for going on to the closing runs were set: oracle plan accuracy not lower than in the first round, named false alarms not above 6 of 27, and no failed run. The first and the third held. The second did not (8 of 27 in `recommended`, 7 of 27 in `no_questions`). That condition contradicted the principle in the same instruction: a legitimate feature whose name does not say plainly that it is known before the outcome is a column of unclear meaning, so the principle drops more of them by construction, and a rule that asks for fewer false alarms while the default moves toward dropping cannot be met by the change it was written for. The condition stands as violated; it was not relaxed or reinterpreted afterwards. The owner decided to keep the principle on purpose and to go on to the closing runs. The record, written before them, is `docs/eval-d-round2.md`.

**What it cost, found before the closing runs** (dev half, drops out of 3 repeats, first round to second round; "recommended" and "no_questions" are the modes where nobody corrects the agent):

| legitimate feature | level | variant | recommended | no_questions |
| --- | --- | --- | --- | --- |
| `humidity_index` | 0.75 | anonymous | 0/3 to 3/3 | 0/3 to 3/3 |
| | | named | 0/3 to 3/3 | 0/3 to 2/3 |
| `deposit_policy` | 0.97 | anonymous | 3/3 to 3/3 | 3/3 to 3/3 |
| | | named | 2/3 to 2/3 | 2/3 to 2/3 |
| `leaf_spot_pattern` | 0.97 | anonymous | 3/3 to 3/3 | 3/3 to 3/3 |
| | | named | 3/3 to 3/3 | 3/3 to 3/3 |
| `employee_position_title` (employee-salaries, OpenML) | 0.75 | n/a | 0/3 to 2/3 | n/a |

`humidity_index` is new in the second round; the others were already dropped in the first. In the same employee-salaries check the leak `2016_gross_pay_received` was dropped in 3 of 3 runs (it was kept in 2 of 3 before), and `underfilled_job_title` was kept in 3 of 3. Source: `results/20261018-123606-gpt-6-luna-employee-salaries-round1` and `...185240-...-round2`.

**The same price on the test half** (named variant, `recommended`, drops out of 3 repeats): `gunes_isinimi` (0.93) 3/3, `odunc_sinifi` (0.97) 3/3, `yenileme_sayisi` (0.85) 3/3, `days_of_cover` (0.93) 2/3, `tahmini_varis_dk` (0.85) 2/3, and `launch_price`, `seat_count`, `alim_noktasi_tipi`, `reorder_gap` 0/3. In the anonymous variant every legitimate feature at 0.75 or more is dropped (24 of 27 occurrences); only the 0.6 level stays, because it is not marked.

## 6. Closing results

The closing runs were made once, with no code change between them and the second round (`1402ac1`): the test half of set D (144 runs), set C with the oracle (192 runs) and the OpenML benchmark (27 agent runs). Whatever came out is reported here. Cost: $0.215, $0.253 and $0.030; no failed run.

### 6.1 Set D, test half

Folder `planted-results/20261018-200114-set-d-test-final-code-gpt-6-luna-3x`. 6 datasets × 3 repeats = 18 runs per row; false alarms out of 27.

| variant | mode | detection | precision | recall | plan accuracy | false alarms | questions/run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anonymous | recommended | 0.87 | 0.78 | 0.90 | 0.87 | 24 | 4.5 |
| named | recommended | 0.94 | 0.87 | 0.94 | 0.92 | 13 | 1.9 |
| anonymous | no_questions | 0.85 | 0.78 | 0.90 | 0.87 | 24 | 0 |
| named | no_questions | 0.94 | 0.87 | 0.94 | 0.92 | 13 | 0 |
| anonymous | not_recommended | 0.56 | 1.00 | 0.55 | 0.83 | 0 | 4.7 |
| named | not_recommended | 0.73 | 1.00 | 0.72 | 0.90 | 0 | 1.8 |
| anonymous | oracle | 0.89 | 1.00 | 0.88 | 0.96 | 0 | 4.6 |
| named | oracle | 0.94 | 0.97 | 0.94 | 0.96 | 3 | 2.1 |

Leaks found by level of association (out of 9 per cell) and the questions asked: in `recommended` the agent found all leaks at 0.75, 0.85 and 0.95 in both variants (27 of 27), 0 of 9 at 0.6 in the anonymous variant and 3 of 9 in the named one. In the anonymous variant it asked about every marked column (24 of 24 at 0.9 or more, 27 of 27 between 0.6 and 0.9); in the named variant about 4 of 9 marked leaks and 7 of 15 marked legitimate features at 0.9 or more, because a name that settles the matter is decided without a question. With the oracle, anonymous: 26 of 27 at the three upper levels, 0 of 9 at 0.6.

**Expectation against result** (`docs/eval-d-test-expectations.md`):

| expectation | result |
| --- | --- |
| oracle plan accuracy, anonymous 0.95 to 0.99 | 0.96, held |
| oracle plan accuracy, named 0.97 to 1.00; under 0.95 would show the dev half overstated it | 0.96: 0.01 under the range, not under 0.95 |
| false alarms, anonymous, `recommended` and `no_questions`: 21 to 24 of 27 | 24, held (top of the range) |
| false alarms, named, same modes: 4 to 9 of 27 | **13, worse than expected** |
| false alarms with the oracle: 0 to 1 anonymous, 0 to 2 named | 0 and **3** |
| leaks at 0.75, 0.85, 0.95 found, anonymous: 25 to 27 of 27 | 27, held |
| leaks at 0.6 found, anonymous: 0 to 3 of 9 | 0, held |
| drop precision, anonymous: 0.72 to 0.82 (`recommended`), 0.90 to 1.00 (`not_recommended`) | 0.78 and 1.00, held |
| questions per run: anonymous 5 to 7, named 1.3 to 2.2 | **4.5 to 4.7**, named 1.8 to 2.1 |
| named variant keeps the legitimate features whose names read as plain facts (4 predicted: `launch_price`, `gunes_isinimi`, `seat_count`, `days_of_cover`) | 2 of the 4 kept (`launch_price`, `seat_count`); `gunes_isinimi` and `days_of_cover` were dropped. Of the 4 predicted dropped, `alim_noktasi_tipi` was kept (named, `recommended` and `no_questions` together: 1 drop in 6) |
| cost $0.22 to $0.30, no failed run | $0.215, no failed run |

None of the four listed signs that the expectations were wrong occurred. Two results matter most for the reading. The named false alarms are higher than on the dev half (13 of 27 against 8 of 27), and the data cannot say whether that is the names or the sample: with 9 legitimate columns per half, one column dropped in every repeat moves the rate by 11 points. And the guess about which names read as plain facts was right for 5 of the 8 columns.

### 6.2 Set C

Folder `planted-results/20261018-200113-set-c-closing-final-code-gpt-6-luna-3x` (the oracle is new in this run; the earlier run is `...093035-set-c-final-code-...`). Set C has no legitimate strong columns, so there is no false alarm rate. Precision is 1.00 in every row.

| variant | mode | detection, earlier run | detection, closing run | plan accuracy, closing | questions/run |
| --- | --- | --- | --- | --- | --- |
| anonymous | recommended | 0.90 | 0.97 | 1.00 | 2.3 |
| named | recommended | 1.00 | 1.00 | 1.00 | 1.1 |
| anonymous | no_questions | 0.92 | 0.93 | 1.00 | 0 |
| named | no_questions | 0.97 | 1.00 | 1.00 | 0 |
| anonymous | not_recommended | 0.62 | 0.68 | 0.89 | 2.2 |
| named | not_recommended | 0.75 | 0.76 | 0.91 | 1.2 |
| anonymous | oracle | n/a | 1.00 | 1.00 | 2.2 |
| named | oracle | n/a | 1.00 | 1.00 | 1.1 |

The gain in the anonymous `recommended` row comes from the regression leaks that are a four-level category of the target (`leak_categorical`); they score 0.74 to 0.80 (`results.json`, `marked`), so the lower tier now marks them, the agent asks and drops them (the earlier run found 6 of 12). The anonymous `no_questions` run finds the `numeric_category` codes in 3 of 12 cases (an integer code that should be cast to a category), against 8 of 12 in the earlier run.

### 6.3 OpenML benchmark

Folder `results/20261018-200117-gpt-6-luna-closing`, compared with the three earlier runs of the same nine datasets, all recomputed with the current noise rule (`summary-new-noise-rule.md` in the older folders; the 16 October table in benchmark.md uses the first rule). "Against the reference" and "against the naive plan" give agent better / worse / within noise. Datasets where the naive plan kept a leak are left out of the comparison with it, unless an agent run kept the leak too, which counts as a loss.

| run | code | valid plans | first plan had errors | questions/run | cost/run | leaky columns kept (of 9) | against the reference | against the naive plan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 Oct | before the signals | 27/27 | 4 | 1.2 | $0.0017 | 0 | 0 / 0 / 9 | 0 / 1 / 6 |
| 18 Oct | with the signals (`d584967`) | 27/27 | 3 | 0.9 | $0.0012 | 1 | 0 / 1 / 8 | 0 / 2 / 6 |
| 18 Oct | after the prompt rule (`25418da`) | 27/27 | 3 | 1.0 | $0.0013 | 1 | 0 / 1 / 8 | 0 / 2 / 6 |
| **18 Oct, closing** | `1402ac1` | **27/27** | 4 | **0.3** | **$0.0011** | **0** | **2 / 0 / 7** | **0 / 1 / 6** |

The "worse" against the reference in rows two and three is the dataset where a run kept `2016_gross_pay_received`. In the closing run `2016_gross_pay_received` was dropped in 3 of 3 runs, `boat` and `body` in 6 of 6, and `underfilled_job_title` was kept in 3 of 3.

Two datasets scored higher than the reference: adult (+0.006) and employee-salaries (+0.033, random forest R² 0.826 against 0.792). That should not be read as the agent preparing the data better. The agent one-hot encodes with more categories than the reference (employee-salaries: every category of `division`, `employee_position_title` and `underfilled_job_title`, against the reference's limits of 50, 100 and 10), and the runs do not isolate whether that or something else moves the score. On this dataset the score moves with the choices of the run and with whether the leak is kept:

| employee-salaries run | forest R² (reference 0.792) | `2016_gross_pay_received` kept | `employee_position_title` dropped |
| --- | --- | --- | --- |
| 16 Oct | 0.812 | 0 of 3 | 0 of 3 |
| 18 Oct, signals | 0.861 † | 1 of 3 | 0 of 3 |
| 18 Oct, after the prompt rule | 0.868 † | 1 of 3 | 0 of 3 |
| first round of the dev corrections | 0.922 † | 2 of 3 | 0 of 3 |
| second round (`1402ac1`) | 0.685 | 0 of 3 | 2 of 3 |
| closing (`1402ac1`) | 0.826 | 0 of 3 | 0 of 3 |

† the score is inflated by a kept leak. The same code (`1402ac1`) gave 0.685 in the second-round check and 0.826 in the closing run, two separate batches of three runs. The difference is whether `employee_position_title` was dropped: `docs/eval-d-test-expectations.md` expected it dropped in 1 to 3 of 3 runs, with a loss of 0.05 to 0.15 of R², and in the closing run it was kept in all three. The question count in the closing run is low (0.3 per run; 9 questions in 27 runs: credit-g 2, telco 1, employee-salaries 6); titanic and house-prices ran without a question, `boat` and `body` being decided by their names.

## 7. What is still limited

- **Leaks that score under 0.6 are not marked.** The planted sets have a few at 0.52 to 0.59. The threshold was not lowered for them: the lower tier is for attention, and a score that low is what an ordinary predictor reaches. They are found only by name (0 of 9 anonymous and 3 of 9 named on the test half).
- **A user who accepts every recommendation, or an agent that may not ask, gets false alarms.** On the test half 24 of 27 (anonymous) and 13 of 27 (named) legitimate strong features were dropped. That is what "drop when unclear" does, by design. The answer to it is the question the agent asks and the report of what was dropped; the measurement is the oracle row, which assumes the user knows.
- **The oracle is an upper bound.** It answers from the ground truth. A real user may not know whether a column is available at prediction time.
- **One model, three repeats, one sample of data per set.** Only gpt-6-luna was run on sets B to D and on the closing runs (gpt-6.1-sol only once, on set A, with one repeat). The test half has 6 datasets, 9 legitimate strong columns and 12 partial leaks; a single column moves a rate by 8 to 11 points. The spread between repeats is small in these tables because the agent often makes the same choices; that does not make the rates precise.
- **The data is synthetic.** The levels are made by recipes and checked on made-up targets, the names are written by us, and what a column means is declared by the recipe. Real columns are messier, and the names in the named variant are not independent of how the prompt reads.
- **Only the test half of set D was untouched until its run.** The dev half was looked at while correcting, and the second-round code was kept after its transition condition was violated (section 5). Set C had been run and read once before its closing run, and it shares its recipes with A and B.
- **The association score is computed before the split.** It uses all rows (at most 20,000, a fixed sample), so labels of later test rows have a small influence on which columns are marked. The effect is small (see the module docstring of `profiler.py`) but not zero.
- **The noise rule changed during the work.** The first benchmark table uses the fold-spread rule, later runs the paired rule. Both are given for the 16 October run.
- **Scores are on the train split** (cross-validation); the test split was never used to compare plans, so none of this says how a model trained on the output does on held-out data.

## Where the numbers come from

All folders are under `docs/benchmark/`. `CODE.txt` in each folder names the code and data commit.

| folder | what it holds |
| --- | --- |
| `planted-results/20261018-024800-gpt-6-luna`, `...030643-gpt-6-luna-3x`, `...030643-gpt-6.1-sol` | set A, before the signals (1 repeat; 3 repeats; the other model, 1 repeat) |
| `planted-results/20261018-081346-set-b-old-code-...`, `...081347-set-b-new-code-...` | set B without and with the signals |
| `planted-results/20261018-093035-set-c-final-code-...` | set C, code `25418da` |
| `planted-results/20261018-111329-set-d-dev-final-code-...`, `...121249-set-d-dev-old-code-oracle-...` | set D dev half, before any correction (three modes; the oracle apart) |
| `planted-results/20261018-123611-set-d-dev-round1-...`, `...185241-set-d-dev-round2-...` | set D dev half after the first and the second round |
| `planted-results/20261018-200114-set-d-test-final-code-...` | set D test half, the one run |
| `planted-results/20261018-200113-set-c-closing-final-code-...` | set C, closing run with the oracle |
| `results/20261016-115230-gpt-6-luna` | OpenML benchmark, 16 October |
| `results/20261018-081350-gpt-6-luna-signals`, `...093038-gpt-6-luna-final` | OpenML benchmark with the signals, and after the prompt rule |
| `results/20261018-123606-...-employee-salaries-round1`, `...185240-...-round2` | employee-salaries only, after the first and the second round |
| `results/20261018-200117-gpt-6-luna-closing` | OpenML benchmark, closing run |

The expectations written before the runs are `docs/eval-d-expectations.md` and `docs/eval-d-test-expectations.md`, and the record of the second round is `docs/eval-d-round2.md`. The closing runs cost $0.498 in all; the first and second round on the dev half cost $0.247 and $0.239.
