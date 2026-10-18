# Planted-fault evaluation

Started 2026-10-18 11:13 UTC, finished 11:55 UTC.
Models: gpt-6-luna. Variants: anonymous, named. Modes: recommended, not_recommended, no_questions. Repeats: 3. Datasets: 6. Runs recorded: 108.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | must_keep dropped | legit strong dropped | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 18 (18) | 0.61 (0.61-0.61) | 0.79 | 0.66 | 15 (0.83/run) | 15 of 27 | 3 (0.17/run) | 4.4 | 2.1 | 0.0019 | 0.0338 |
| gpt-6-luna | named | recommended | 18 (18) | 0.86 (0.84-0.87) | 0.83 | 0.85 | 15 (0.83/run) | 15 of 27 | 0 (0.00/run) | 2.6 | 2.4 | 0.0019 | 0.0349 |
| gpt-6-luna | anonymous | not_recommended | 18 (18) | 0.48 (0.42-0.55) | 1.00 | 0.46 | 0 (0.00/run) | 0 of 27 | 0 (0.00/run) | 4.4 | 2.2 | 0.0018 | 0.0321 |
| gpt-6-luna | named | not_recommended | 18 (18) | 0.55 (0.52-0.58) | 1.00 | 0.52 | 0 (0.00/run) | 0 of 27 | 0 (0.00/run) | 2.2 | 2.1 | 0.0017 | 0.0298 |
| gpt-6-luna | anonymous | no_questions | 18 (18) | 0.61 (0.61-0.61) | 0.79 | 0.66 | 15 (0.83/run) | 15 of 27 | 1 (0.06/run) | 0.0 | 1.1 | 0.0010 | 0.0177 |
| gpt-6-luna | named | no_questions | 18 (18) | 0.76 (0.71-0.84) | 0.81 | 0.75 | 15 (0.83/run) | 15 of 27 | 0 (0.00/run) | 0.0 | 1.0 | 0.0009 | 0.0159 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel | leak_partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 33% (3/9) | 0% (0/6) | 100% (6/6) | not scored | 33% (12/36) |
| gpt-6-luna | named | recommended | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 33% (3/9) | 100% (6/6) | 100% (6/6) | not scored | 81% (29/36) |
| gpt-6-luna | anonymous | not_recommended | n/a | n/a | n/a | 67% (6/9) | 100% (9/9) | 100% (6/6) | 33% (1/3) | 100% (6/6) | 100% (3/3) | 33% (3/9) | 83% (5/6) | 100% (6/6) | not scored | 0% (0/36) |
| gpt-6-luna | named | not_recommended | n/a | n/a | n/a | 78% (7/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 33% (3/9) | 100% (6/6) | 100% (6/6) | not scored | 6% (2/36) |
| gpt-6-luna | anonymous | no_questions | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 33% (3/9) | 0% (0/6) | 100% (6/6) | not scored | 33% (12/36) |
| gpt-6-luna | named | no_questions | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 33% (3/9) | 100% (6/6) | 100% (6/6) | not scored | 56% (20/36) |

## Leaks of partial strength found, by level of association

Found out of planted, over all ok runs of the group. The level is the association the column was made to have; what makes the column a leak is what it means.

| model | variant | mode | 0.60 | 0.75 | 0.85 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 3/9 | 0/9 | 0/9 | 9/9 |
| gpt-6-luna | named | recommended | 9/9 | 5/9 | 6/9 | 9/9 |
| gpt-6-luna | anonymous | not_recommended | 0/9 | 0/9 | 0/9 | 0/9 |
| gpt-6-luna | named | not_recommended | 0/9 | 2/9 | 0/9 | 0/9 |
| gpt-6-luna | anonymous | no_questions | 3/9 | 0/9 | 0/9 | 9/9 |
| gpt-6-luna | named | no_questions | 4/9 | 2/9 | 5/9 | 9/9 |

## Legitimate strong features dropped, by level of association

Dropped out of planted, over all ok runs of the group. Every drop is a false alarm.

| model | variant | mode | 0.60 | 0.75 | 0.85 | 0.93 | 0.97 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 0/3 | 0/3 | 0/6 | 9/9 | 6/6 |
| gpt-6-luna | named | recommended | 0/3 | 0/3 | 0/6 | 9/9 | 6/6 |
| gpt-6-luna | anonymous | not_recommended | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |
| gpt-6-luna | named | not_recommended | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |
| gpt-6-luna | anonymous | no_questions | 0/3 | 0/3 | 0/6 | 9/9 | 6/6 |
| gpt-6-luna | named | no_questions | 0/3 | 0/3 | 0/6 | 9/9 | 6/6 |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | recommended | 0.61 | 0.86 | +25 pp | 0.79 | 0.83 | +4 pp |
| gpt-6-luna | not_recommended | 0.48 | 0.55 | +6 pp | 1.00 | 1.00 | +0 pp |
| gpt-6-luna | no_questions | 0.61 | 0.76 | +15 pp | 0.79 | 0.81 | +2 pp |

## Notes

- Questions asked in all runs: 246. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
