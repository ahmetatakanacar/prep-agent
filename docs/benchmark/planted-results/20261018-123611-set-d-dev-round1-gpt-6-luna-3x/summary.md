# Planted-fault evaluation

Started 2026-10-18 12:36 UTC, finished 13:38 UTC.
Models: gpt-6-luna. Variants: anonymous, named. Modes: recommended, not_recommended, no_questions, oracle. Repeats: 3. Datasets: 6. Runs recorded: 144.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | plan accuracy | must_keep dropped | legit strong dropped | asked: suspected | asked: high | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 18 (18) | 0.66 (0.65-0.68) | 0.80 | 0.69 | 0.85 | 15 (0.83/run) | 15 of 27 | 24 of 24 | 30 of 30 | 2 (0.11/run) | 6.1 | 2.1 | 0.0021 | 0.0379 |
| gpt-6-luna | named | recommended | 18 (18) | 0.82 (0.81-0.84) | 0.92 | 0.80 | 0.92 | 6 (0.33/run) | 6 of 27 | 11 of 24 | 17 of 30 | 0 (0.00/run) | 1.7 | 2.4 | 0.0021 | 0.0371 |
| gpt-6-luna | anonymous | not_recommended | 18 (18) | 0.73 (0.71-0.77) | 0.85 | 0.72 | 0.87 | 11 (0.61/run) | 9 of 27 | 24 of 24 | 30 of 30 | 3 (0.17/run) | 6.2 | 2.3 | 0.0021 | 0.0378 |
| gpt-6-luna | named | not_recommended | 18 (18) | 0.84 (0.77-0.87) | 0.97 | 0.83 | 0.94 | 2 (0.11/run) | 2 of 27 | 9 of 24 | 18 of 30 | 0 (0.00/run) | 1.7 | 2.0 | 0.0017 | 0.0303 |
| gpt-6-luna | anonymous | no_questions | 18 (18) | 0.65 (0.65-0.65) | 0.80 | 0.69 | 0.85 | 15 (0.83/run) | 15 of 27 | n/a | n/a | 0 (0.00/run) | 0.0 | 1.0 | 0.0010 | 0.0177 |
| gpt-6-luna | named | no_questions | 18 (18) | 0.89 (0.87-0.94) | 0.93 | 0.89 | 0.94 | 6 (0.33/run) | 6 of 27 | n/a | n/a | 0 (0.00/run) | 0.0 | 1.0 | 0.0009 | 0.0168 |
| gpt-6-luna | anonymous | oracle | 18 (18) | 0.92 (0.90-0.94) | 1.00 | 0.92 | 0.97 | 0 (0.00/run) | 0 of 27 | 24 of 24 | 30 of 30 | 0 (0.00/run) | 6.3 | 2.2 | 0.0020 | 0.0369 |
| gpt-6-luna | named | oracle | 18 (18) | 0.97 (0.94-1.00) | 1.00 | 0.97 | 0.99 | 0 (0.00/run) | 0 of 27 | 12 of 24 | 16 of 30 | 0 (0.00/run) | 1.8 | 2.2 | 0.0018 | 0.0326 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel | leak_partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 17% (1/6) | 100% (6/6) | not scored | 25% (9/36) |
| gpt-6-luna | named | recommended | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 53% (19/36) |
| gpt-6-luna | anonymous | not_recommended | n/a | n/a | n/a | 44% (4/9) | 100% (9/9) | 100% (6/6) | 0% (0/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 83% (5/6) | 100% (6/6) | not scored | 56% (20/36) |
| gpt-6-luna | named | not_recommended | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 58% (21/36) |
| gpt-6-luna | anonymous | no_questions | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 0% (0/6) | 100% (6/6) | not scored | 25% (9/36) |
| gpt-6-luna | named | no_questions | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 72% (26/36) |
| gpt-6-luna | anonymous | oracle | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 81% (29/36) |
| gpt-6-luna | named | oracle | n/a | n/a | n/a | 100% (9/9) | 100% (9/9) | 100% (6/6) | 100% (3/3) | 100% (6/6) | 100% (3/3) | 100% (9/9) | 100% (6/6) | 100% (6/6) | not scored | 92% (33/36) |

## Leaks of partial strength found, by level of association

Found out of planted, over all ok runs of the group. The level is the association the column was made to have; what makes the column a leak is what it means.

| model | variant | mode | 0.60 | 0.75 | 0.85 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 0/9 | 0/9 | 0/9 | 9/9 |
| gpt-6-luna | named | recommended | 7/9 | 1/9 | 2/9 | 9/9 |
| gpt-6-luna | anonymous | not_recommended | 2/9 | 9/9 | 9/9 | 0/9 |
| gpt-6-luna | named | not_recommended | 3/9 | 7/9 | 7/9 | 4/9 |
| gpt-6-luna | anonymous | no_questions | 0/9 | 0/9 | 0/9 | 9/9 |
| gpt-6-luna | named | no_questions | 7/9 | 4/9 | 6/9 | 9/9 |
| gpt-6-luna | anonymous | oracle | 2/9 | 9/9 | 9/9 | 9/9 |
| gpt-6-luna | named | oracle | 8/9 | 8/9 | 8/9 | 9/9 |

## Legitimate strong features dropped, by level of association

Dropped out of planted, over all ok runs of the group. Every drop is a false alarm.

| model | variant | mode | 0.60 | 0.75 | 0.85 | 0.93 | 0.97 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 0/3 | 0/3 | 0/6 | 9/9 | 6/6 |
| gpt-6-luna | named | recommended | 0/3 | 0/3 | 0/6 | 1/9 | 5/6 |
| gpt-6-luna | anonymous | not_recommended | 0/3 | 3/3 | 6/6 | 0/9 | 0/6 |
| gpt-6-luna | named | not_recommended | 0/3 | 2/3 | 0/6 | 0/9 | 0/6 |
| gpt-6-luna | anonymous | no_questions | 0/3 | 0/3 | 0/6 | 9/9 | 6/6 |
| gpt-6-luna | named | no_questions | 0/3 | 0/3 | 0/6 | 1/9 | 5/6 |
| gpt-6-luna | anonymous | oracle | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |
| gpt-6-luna | named | oracle | 0/3 | 0/3 | 0/6 | 0/9 | 0/6 |

## Questions about columns strongly related to the target

Columns that a question named, out of those the profile related to the target at 0.6 or more (0.9 or more is `suspected`), by what the ground truth says the column is. The no_questions mode cannot ask.

| model | variant | mode | leaks, 0.9 and over | legit, 0.9 and over | leaks, 0.6 to 0.9 | legit, 0.6 to 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 9 of 9 | 15 of 15 | 21 of 21 | 9 of 9 |
| gpt-6-luna | named | recommended | 6 of 9 | 5 of 15 | 16 of 21 | 1 of 9 |
| gpt-6-luna | anonymous | not_recommended | 9 of 9 | 15 of 15 | 21 of 21 | 9 of 9 |
| gpt-6-luna | named | not_recommended | 5 of 9 | 4 of 15 | 15 of 21 | 3 of 9 |
| gpt-6-luna | anonymous | no_questions | n/a | n/a | n/a | n/a |
| gpt-6-luna | named | no_questions | n/a | n/a | n/a | n/a |
| gpt-6-luna | anonymous | oracle | 9 of 9 | 15 of 15 | 21 of 21 | 9 of 9 |
| gpt-6-luna | named | oracle | 7 of 9 | 5 of 15 | 15 of 21 | 1 of 9 |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | recommended | 0.66 | 0.82 | +16 pp | 0.80 | 0.92 | +12 pp |
| gpt-6-luna | not_recommended | 0.73 | 0.84 | +11 pp | 0.85 | 0.97 | +12 pp |
| gpt-6-luna | no_questions | 0.65 | 0.89 | +25 pp | 0.80 | 0.93 | +13 pp |
| gpt-6-luna | oracle | 0.92 | 0.97 | +4 pp | 1.00 | 1.00 | +0 pp |

## Notes

- Questions asked in all runs: 430. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
