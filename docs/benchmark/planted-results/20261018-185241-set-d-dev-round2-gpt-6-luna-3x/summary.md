# Planted-fault evaluation

Started 2026-10-18 18:52 UTC, finished 19:56 UTC.
Models: gpt-6-luna. Variants: anonymous, named. Modes: recommended, not_recommended, no_questions, oracle. Repeats: 3. Datasets: 6. Runs recorded: 144.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | plan accuracy | must_keep dropped | legit strong dropped | asked: suspected | asked: high | kept by decision | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 18 (18) | 0.87 (0.87-0.87) | 0.77 | 0.93 | 0.89 | 24 (1.33/run) | 24 of 27 | 24 of 24 | 30 of 30 | 0 | 1 (0.06/run) | 6.3 | 2.4 | 0.0022 | 0.0397 |
| gpt-6-luna | named | recommended | 18 (18) | 0.96 (0.94-0.97) | 0.91 | 0.95 | 0.96 | 8 (0.44/run) | 8 of 27 | 11 of 24 | 14 of 30 | 0 | 0 (0.00/run) | 1.6 | 2.1 | 0.0018 | 0.0321 |
| gpt-6-luna | anonymous | not_recommended | 18 (18) | 0.52 (0.52-0.52) | 0.98 | 0.48 | 0.83 | 1 (0.06/run) | 0 of 27 | 24 of 24 | 30 of 30 | 54 | 3 (0.17/run) | 6.2 | 2.2 | 0.0020 | 0.0365 |
| gpt-6-luna | named | not_recommended | 18 (18) | 0.76 (0.71-0.81) | 0.98 | 0.75 | 0.92 | 1 (0.06/run) | 0 of 27 | 9 of 24 | 15 of 30 | 18 | 0 (0.00/run) | 1.4 | 1.9 | 0.0016 | 0.0284 |
| gpt-6-luna | anonymous | no_questions | 18 (18) | 0.85 (0.84-0.87) | 0.77 | 0.91 | 0.88 | 24 (1.33/run) | 24 of 27 | n/a | n/a | 0 | 0 (0.00/run) | 0.0 | 1.1 | 0.0010 | 0.0179 |
| gpt-6-luna | named | no_questions | 18 (18) | 0.99 (0.97-1.00) | 0.92 | 0.99 | 0.97 | 7 (0.39/run) | 7 of 27 | n/a | n/a | 0 | 0 (0.00/run) | 0.0 | 1.0 | 0.0009 | 0.0156 |
| gpt-6-luna | anonymous | oracle | 18 (18) | 0.94 (0.94-0.94) | 1.00 | 0.93 | 0.98 | 0 (0.00/run) | 0 of 27 | 24 of 24 | 30 of 30 | 24 | 0 (0.00/run) | 6.2 | 2.4 | 0.0022 | 0.0387 |
| gpt-6-luna | named | oracle | 18 (18) | 0.97 (0.94-1.00) | 0.99 | 0.97 | 0.99 | 1 (0.06/run) | 1 of 27 | 10 of 24 | 19 of 30 | 7 | 0 (0.00/run) | 1.7 | 2.1 | 0.0017 | 0.0301 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel | leak_partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 0% (0/6) | 100% (6/6) | not scored | 83% (30/36) |
| gpt-6-luna | named | recommended | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 89% (32/36) |
| gpt-6-luna | anonymous | not_recommended | n/a | n/a | n/a | 33% (3/9) | 100% (9/9) | 100% (6/6) | 0% (0/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 0% (0/36) |
| gpt-6-luna | named | not_recommended | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 39% (14/36) |
| gpt-6-luna | anonymous | no_questions | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 33% (1/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 0% (0/6) | 100% (6/6) | not scored | 83% (30/36) |
| gpt-6-luna | named | no_questions | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 97% (35/36) |
| gpt-6-luna | anonymous | oracle | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 83% (30/36) |
| gpt-6-luna | named | oracle | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 92% (33/36) |

## Leaks of partial strength found, by level of association

Found out of planted, over all ok runs of the group. The level is the association the column was made to have; what makes the column a leak is what it means.

| model | variant | mode | 0.60 | 0.75 | 0.85 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 3/9 | 9/9 | 9/9 | 9/9 |
| gpt-6-luna | named | recommended | 5/9 | 9/9 | 9/9 | 9/9 |
| gpt-6-luna | anonymous | not_recommended | 0/9 | 0/9 | 0/9 | 0/9 |
| gpt-6-luna | named | not_recommended | 5/9 | 1/9 | 5/9 | 3/9 |
| gpt-6-luna | anonymous | no_questions | 3/9 | 9/9 | 9/9 | 9/9 |
| gpt-6-luna | named | no_questions | 8/9 | 9/9 | 9/9 | 9/9 |
| gpt-6-luna | anonymous | oracle | 3/9 | 9/9 | 9/9 | 9/9 |
| gpt-6-luna | named | oracle | 6/9 | 9/9 | 9/9 | 9/9 |

## Legitimate strong features dropped, by level of association

Dropped out of planted, over all ok runs of the group. Every drop is a false alarm.

| model | variant | mode | 0.60 | 0.75 | 0.85 | 0.93 | 0.97 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 0/3 | 3/3 | 6/6 | 9/9 | 6/6 |
| gpt-6-luna | named | recommended | 0/3 | 3/3 | 0/6 | 0/9 | 5/6 |
| gpt-6-luna | anonymous | not_recommended | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |
| gpt-6-luna | named | not_recommended | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |
| gpt-6-luna | anonymous | no_questions | 0/3 | 3/3 | 6/6 | 9/9 | 6/6 |
| gpt-6-luna | named | no_questions | 0/3 | 2/3 | 0/6 | 0/9 | 5/6 |
| gpt-6-luna | anonymous | oracle | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |
| gpt-6-luna | named | oracle | 0/3 | 0/3 | 0/6 | 0/9 | 1/6 |

## Questions about columns strongly related to the target

Columns that a question named, out of those the profile related to the target at 0.6 or more (0.9 or more is `suspected`), by what the ground truth says the column is. The no_questions mode cannot ask.

| model | variant | mode | leaks, 0.9 and over | legit, 0.9 and over | leaks, 0.6 to 0.9 | legit, 0.6 to 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 9 of 9 | 15 of 15 | 21 of 21 | 9 of 9 |
| gpt-6-luna | named | recommended | 6 of 9 | 5 of 15 | 11 of 21 | 3 of 9 |
| gpt-6-luna | anonymous | not_recommended | 9 of 9 | 15 of 15 | 21 of 21 | 9 of 9 |
| gpt-6-luna | named | not_recommended | 6 of 9 | 3 of 15 | 12 of 21 | 3 of 9 |
| gpt-6-luna | anonymous | no_questions | n/a | n/a | n/a | n/a |
| gpt-6-luna | named | no_questions | n/a | n/a | n/a | n/a |
| gpt-6-luna | anonymous | oracle | 9 of 9 | 15 of 15 | 21 of 21 | 9 of 9 |
| gpt-6-luna | named | oracle | 6 of 9 | 4 of 15 | 16 of 21 | 3 of 9 |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | recommended | 0.87 | 0.96 | +9 pp | 0.77 | 0.91 | +14 pp |
| gpt-6-luna | not_recommended | 0.52 | 0.76 | +25 pp | 0.98 | 0.98 | +1 pp |
| gpt-6-luna | no_questions | 0.85 | 0.99 | +14 pp | 0.77 | 0.92 | +16 pp |
| gpt-6-luna | oracle | 0.94 | 0.97 | +3 pp | 1.00 | 0.99 | -1 pp |

## Notes

- Questions asked in all runs: 420. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
