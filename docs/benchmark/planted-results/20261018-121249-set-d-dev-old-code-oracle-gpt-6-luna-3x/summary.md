# Planted-fault evaluation

Started 2026-10-18 12:12 UTC, finished 12:30 UTC.
Models: gpt-6-luna. Variants: anonymous, named. Modes: oracle. Repeats: 3. Datasets: 6. Runs recorded: 36.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | plan accuracy | must_keep dropped | legit strong dropped | asked: suspected | asked: high | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | oracle | 18 (18) | 0.67 (0.65-0.68) | 1.00 | 0.66 | 0.89 | 0 (0.00/run) | 0 of 27 | 24 of 24 | 3 of 30 | 1 (0.06/run) | 4.4 | 2.4 | 0.0020 | 0.0361 |
| gpt-6-luna | named | oracle | 18 (18) | 0.88 (0.87-0.90) | 1.00 | 0.87 | 0.96 | 0 (0.00/run) | 0 of 27 | 24 of 24 | 16 of 30 | 0 (0.00/run) | 2.6 | 2.3 | 0.0018 | 0.0330 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel | leak_partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | oracle | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 33% (3/9) | 83% (5/6) | 100% (6/6) | not scored | 33% (12/36) |
| gpt-6-luna | named | oracle | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 33% (3/9) | 100% (6/6) | 100% (6/6) | not scored | 86% (31/36) |

## Leaks of partial strength found, by level of association

Found out of planted, over all ok runs of the group. The level is the association the column was made to have; what makes the column a leak is what it means.

| model | variant | mode | 0.60 | 0.75 | 0.85 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | oracle | 3/9 | 0/9 | 0/9 | 9/9 |
| gpt-6-luna | named | oracle | 9/9 | 5/9 | 8/9 | 9/9 |

## Legitimate strong features dropped, by level of association

Dropped out of planted, over all ok runs of the group. Every drop is a false alarm.

| model | variant | mode | 0.60 | 0.75 | 0.85 | 0.93 | 0.97 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | oracle | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |
| gpt-6-luna | named | oracle | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |

## Questions about columns strongly related to the target

Columns that a question named, out of those the profile related to the target at 0.6 or more (0.9 or more is `suspected`), by what the ground truth says the column is. The no_questions mode cannot ask.

| model | variant | mode | leaks, 0.9 and over | legit, 0.9 and over | leaks, 0.6 to 0.9 | legit, 0.6 to 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | oracle | 9 of 9 | 15 of 15 | 3 of 21 | 0 of 9 |
| gpt-6-luna | named | oracle | 9 of 9 | 15 of 15 | 16 of 21 | 0 of 9 |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | oracle | 0.67 | 0.88 | +22 pp | 1.00 | 1.00 | +0 pp |

## Notes

- Questions asked in all runs: 127. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
