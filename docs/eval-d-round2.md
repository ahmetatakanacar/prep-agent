# Set D, second round of fixes: what it measured before the closing runs

This is a finding recorded before the closing measurements (D-test, set C and the OpenML benchmark). Nothing in it was changed afterwards. The runs are in `docs/benchmark/planted-results/20261018-123611-set-d-dev-round1-gpt-6-luna-3x/` (first round) and `20261018-185241-set-d-dev-round2-gpt-6-luna-3x/` (second round); the employee-salaries checks are in `docs/benchmark/results/20261018-123606-gpt-6-luna-employee-salaries-round1/` and `20261018-185240-gpt-6-luna-employee-salaries-round2/`. All are gpt-6-luna, three repeats, on the dev half of set D (`planted-d01` to `planted-d06`), both variants, four answer modes.

## What the second round changed

The decision behind it, taken on a principle and not on the dev numbers: keeping a column that leaks the target is a silent error (the scores come out better than the model is, and nobody sees it), and it is the error this tool exists to prevent; dropping a column that was a real feature is a visible loss that can be undone. So:

1. A marked column (`suspected_target_leakage` or `high_target_association`) whose meaning the planner cannot tell is dropped by default, with or without questions: it is the recommended option when the planner asks, and what the planner does when it may not ask. A column whose name plainly says it is known before the outcome is still kept, with the reason written down.
2. A marked column the planner asked about and the user chose to keep is no longer a validator warning; the report says it was kept by the user's decision, with the question and the answer. A marked column that stays without a question keeps its warning.
3. Leaks that score under 0.6 are not marked and the threshold was not lowered for them (a known limit).

## Result against the first round

| variant | mode | detection | precision | recall | plan accuracy | false alarms (legitimate strong features dropped) |
| --- | --- | --- | --- | --- | --- | --- |
| anonymous | recommended | 0.66 to 0.87 | 0.80 to 0.77 | 0.69 to 0.93 | 0.85 to 0.89 | 15/27 to 24/27 |
| named | recommended | 0.82 to 0.96 | 0.92 to 0.91 | 0.80 to 0.95 | 0.92 to 0.96 | 6/27 to 8/27 |
| anonymous | no_questions | 0.65 to 0.85 | 0.80 to 0.77 | 0.69 to 0.91 | 0.85 to 0.88 | 15/27 to 24/27 |
| named | no_questions | 0.89 to 0.99 | 0.93 to 0.92 | 0.89 to 0.99 | 0.94 to 0.97 | 6/27 to 7/27 |
| anonymous | oracle | 0.92 to 0.94 | 1.00 to 1.00 | 0.92 to 0.93 | 0.97 to 0.98 | 0/27 to 0/27 |
| named | oracle | 0.97 to 0.97 | 1.00 to 0.99 | 0.97 to 0.97 | 0.99 to 0.99 | 0/27 to 1/27 |

No run failed (144 of 144 ok in both rounds). Leaks found by level in `recommended`: anonymous 0.75 and 0.85 from 0/9 to 9/9 each, named from 1/9 and 2/9 to 9/9 each; 0.95 stayed at 9/9. The cost of the round was $0.239.

## The transition condition, and why it was violated

Before the second round three conditions for going on to the closing measurements were set: oracle plan accuracy not lower than in the first round (anonymous at least 0.95, named at least 0.97); false alarms in the named variant not worse than in the first round (at most 6 of 27); no failed run. The first and third held. The second did not: named false alarms went from 6/27 to 8/27 in `recommended` and 7/27 in `no_questions`.

**That condition was defined in contradiction with the principle in the same instruction.** The principle drops columns of unclear meaning, and a legitimate feature whose name does not say plainly that it is known before the outcome is a column of unclear meaning; more of them are dropped by construction. A rule that asks for fewer false alarms while the default is moved toward dropping cannot be met by the change it was written for. The condition was therefore violated, and it stands as violated: it was not relaxed or reinterpreted afterwards. The owner decided to keep the principle on purpose and to go on to the closing measurements, with the violation recorded here.

## What the principle cost: the legitimate features that are now dropped

Legitimate features dropped by the agent, out of three repeats, first round and second round. "Recommended" and "no_questions" are the modes where nobody corrects the agent; with the oracle (an informed user) the feature is kept.

| column (set D dev) | level | variant | recommended | no_questions |
| --- | --- | --- | --- | --- |
| `humidity_index` (d05) | 0.75 | anonymous | 0/3 to 3/3 | 0/3 to 3/3 |
| | | named | 0/3 to 3/3 | 0/3 to 2/3 |
| `deposit_policy` (d01) | 0.97 | anonymous | 3/3 to 3/3 | 3/3 to 3/3 |
| | | named | 2/3 to 2/3 | 2/3 to 2/3 |
| `leaf_spot_pattern` (d05) | 0.97 | anonymous | 3/3 to 3/3 | 3/3 to 3/3 |
| | | named | 3/3 to 3/3 | 3/3 to 3/3 |

`humidity_index` is the new loss: a measured field value whose name does not say when it is known. `deposit_policy` and `leaf_spot_pattern` were dropped already in the first round, because their score is above 0.9 and their names do not plainly say "known before the outcome"; the second round did not change that. In the anonymous variant every legitimate feature at 0.75 and above is dropped in these two modes (24 of 27), which is what the principle says to do when nothing but a score is known.

In the employee-salaries benchmark (three runs, second round): `employee_position_title`, a legitimate predictor scored 0.75 that the reference plan keeps, was asked about and dropped in two of three runs. The leak in the same table, `2016_gross_pay_received`, was dropped in all three (two of three kept it in the first round), and `underfilled_job_title` was kept in all three.

## Not changed by the round

Oracle plan accuracy stays at 0.98 (anonymous) and 0.99 (named): when the user is informed, the questions the agent asks lead to a nearly right plan in both variants, and the false alarms of the oracle runs are 0 and 1 of 27.
