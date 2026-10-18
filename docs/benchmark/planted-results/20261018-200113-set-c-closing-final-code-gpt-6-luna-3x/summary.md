# Planted-fault evaluation

Started 2026-10-18 20:01 UTC, finished 21:10 UTC.
Models: gpt-6-luna. Variants: anonymous, named. Modes: recommended, not_recommended, no_questions, oracle. Repeats: 3. Datasets: 8. Runs recorded: 192.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | plan accuracy | must_keep dropped | legit strong dropped | asked: suspected | asked: high | kept by decision | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 24 (24) | 0.97 (0.97-0.97) | 1.00 | 1.00 | 1.00 | 0 (0.00/run) | n/a | 24 of 24 | 6 of 6 | 0 | 6 (0.25/run) | 2.3 | 2.2 | 0.0017 | 0.0402 |
| gpt-6-luna | named | recommended | 24 (24) | 1.00 (1.00-1.00) | 1.00 | 1.00 | 1.00 | 0 (0.00/run) | n/a | 24 of 24 | 3 of 6 | 0 | 0 (0.00/run) | 1.1 | 2.0 | 0.0014 | 0.0328 |
| gpt-6-luna | anonymous | not_recommended | 24 (24) | 0.68 (0.62-0.70) | 1.00 | 0.69 | 0.89 | 0 (0.00/run) | n/a | 24 of 24 | 6 of 6 | 30 | 1 (0.04/run) | 2.2 | 2.1 | 0.0016 | 0.0375 |
| gpt-6-luna | named | not_recommended | 24 (24) | 0.76 (0.75-0.78) | 1.00 | 0.73 | 0.91 | 0 (0.00/run) | n/a | 24 of 24 | 5 of 6 | 27 | 0 (0.00/run) | 1.2 | 2.0 | 0.0014 | 0.0337 |
| gpt-6-luna | anonymous | no_questions | 24 (24) | 0.93 (0.93-0.93) | 1.00 | 1.00 | 1.00 | 0 (0.00/run) | n/a | n/a | n/a | 0 | 6 (0.25/run) | 0.0 | 1.0 | 0.0009 | 0.0218 |
| gpt-6-luna | named | no_questions | 24 (24) | 1.00 (1.00-1.00) | 1.00 | 1.00 | 1.00 | 0 (0.00/run) | n/a | n/a | n/a | 0 | 0 (0.00/run) | 0.0 | 1.0 | 0.0009 | 0.0207 |
| gpt-6-luna | anonymous | oracle | 24 (24) | 1.00 (1.00-1.00) | 1.00 | 1.00 | 1.00 | 0 (0.00/run) | n/a | 24 of 24 | 6 of 6 | 0 | 0 (0.00/run) | 2.2 | 2.1 | 0.0015 | 0.0353 |
| gpt-6-luna | named | oracle | 24 (24) | 1.00 (1.00-1.00) | 1.00 | 1.00 | 1.00 | 0 (0.00/run) | n/a | 24 of 24 | 3 of 6 | 0 | 0 (0.00/run) | 1.1 | 2.0 | 0.0013 | 0.0309 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel | leak_partial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 75% (9/12) | 100% (12/12) | not scored | n/a |
| gpt-6-luna | named | recommended | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 100% (12/12) | 100% (12/12) | not scored | n/a |
| gpt-6-luna | anonymous | not_recommended | 0% (0/9) | 0% (0/9) | 0% (0/12) | 75% (9/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 92% (11/12) | 100% (6/6) | 100% (9/9) | 58% (7/12) | 100% (12/12) | not scored | n/a |
| gpt-6-luna | named | not_recommended | 0% (0/9) | 0% (0/9) | 8% (1/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 100% (12/12) | 100% (12/12) | not scored | n/a |
| gpt-6-luna | anonymous | no_questions | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 25% (3/12) | 100% (12/12) | not scored | n/a |
| gpt-6-luna | named | no_questions | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 100% (12/12) | 100% (12/12) | not scored | n/a |
| gpt-6-luna | anonymous | oracle | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 100% (12/12) | 100% (12/12) | not scored | n/a |
| gpt-6-luna | named | oracle | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 100% (12/12) | 100% (12/12) | not scored | n/a |

## Questions about columns strongly related to the target

Columns that a question named, out of those the profile related to the target at 0.6 or more (0.9 or more is `suspected`), by what the ground truth says the column is. The no_questions mode cannot ask.

| model | variant | mode | leaks, 0.9 and over | legit, 0.9 and over | leaks, 0.6 to 0.9 | legit, 0.6 to 0.9 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 24 of 24 | n/a | 6 of 6 | n/a |
| gpt-6-luna | named | recommended | 24 of 24 | n/a | 3 of 6 | n/a |
| gpt-6-luna | anonymous | not_recommended | 24 of 24 | n/a | 6 of 6 | n/a |
| gpt-6-luna | named | not_recommended | 24 of 24 | n/a | 5 of 6 | n/a |
| gpt-6-luna | anonymous | no_questions | n/a | n/a | n/a | n/a |
| gpt-6-luna | named | no_questions | n/a | n/a | n/a | n/a |
| gpt-6-luna | anonymous | oracle | 24 of 24 | n/a | 6 of 6 | n/a |
| gpt-6-luna | named | oracle | 24 of 24 | n/a | 3 of 6 | n/a |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | recommended | 0.97 | 1.00 | +3 pp | 1.00 | 1.00 | +0 pp |
| gpt-6-luna | not_recommended | 0.68 | 0.76 | +8 pp | 1.00 | 1.00 | +0 pp |
| gpt-6-luna | no_questions | 0.93 | 1.00 | +7 pp | 1.00 | 1.00 | +0 pp |
| gpt-6-luna | oracle | 1.00 | 1.00 | +0 pp | 1.00 | 1.00 | +0 pp |

## Notes

- Questions asked in all runs: 244. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
