# Planted-fault evaluation

Started 2026-10-18 03:06 UTC, finished 03:50 UTC.
Models: gpt-6-luna. Variants: anonymous, named. Modes: recommended, not_recommended, no_questions. Repeats: 3. Datasets: 8. Runs recorded: 144.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | must_keep dropped | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 24 (24) | 0.58 (0.50-0.65) | 1.00 | 0.59 | 0 (0.00/run) | 6 (0.25/run) | 1.3 | 2.1 | 0.0015 | 0.0348 |
| gpt-6-luna | named | recommended | 24 (24) | 0.72 (0.70-0.75) | 0.93 | 0.69 | 6 (0.25/run) | 0 (0.00/run) | 0.8 | 1.6 | 0.0012 | 0.0281 |
| gpt-6-luna | anonymous | not_recommended | 24 (24) | 0.42 (0.38-0.45) | 0.98 | 0.42 | 1 (0.04/run) | 0 (0.00/run) | 1.5 | 2.0 | 0.0014 | 0.0338 |
| gpt-6-luna | named | not_recommended | 24 (24) | 0.62 (0.62-0.62) | 1.00 | 0.58 | 0 (0.00/run) | 0 (0.00/run) | 0.5 | 1.4 | 0.0010 | 0.0230 |
| gpt-6-luna | anonymous | no_questions | 24 (24) | 0.57 (0.55-0.60) | 1.00 | 0.56 | 0 (0.00/run) | 6 (0.25/run) | 0.0 | 1.0 | 0.0008 | 0.0198 |
| gpt-6-luna | named | no_questions | 24 (24) | 0.68 (0.65-0.70) | 0.97 | 0.64 | 2 (0.08/run) | 0 (0.00/run) | 0.0 | 1.0 | 0.0007 | 0.0165 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 0% (0/9) | 0% (0/9) | 0% (0/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 92% (11/12) | 33% (2/6) | 0% (0/9) | 50% (6/12) | 100% (12/12) | not scored |
| gpt-6-luna | named | recommended | 22% (2/9) | 44% (4/9) | 50% (6/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 0% (0/6) | 0% (0/9) | 100% (12/12) | 100% (12/12) | not scored |
| gpt-6-luna | anonymous | not_recommended | 0% (0/9) | 0% (0/9) | 0% (0/12) | 50% (6/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 0% (0/12) | 0% (0/6) | 0% (0/9) | 42% (5/12) | 100% (12/12) | not scored |
| gpt-6-luna | named | not_recommended | 0% (0/9) | 0% (0/9) | 0% (0/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 0% (0/6) | 0% (0/9) | 100% (12/12) | 100% (12/12) | not scored |
| gpt-6-luna | anonymous | no_questions | 0% (0/9) | 0% (0/9) | 0% (0/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 83% (10/12) | 0% (0/6) | 0% (0/9) | 67% (8/12) | 100% (12/12) | not scored |
| gpt-6-luna | named | no_questions | 11% (1/9) | 0% (0/9) | 42% (5/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 0% (0/6) | 0% (0/9) | 100% (12/12) | 100% (12/12) | not scored |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | recommended | 0.58 | 0.72 | +14 pp | 1.00 | 0.93 | -7 pp |
| gpt-6-luna | not_recommended | 0.42 | 0.62 | +21 pp | 0.98 | 1.00 | +2 pp |
| gpt-6-luna | no_questions | 0.57 | 0.68 | +10 pp | 1.00 | 0.97 | -3 pp |

## Notes

- Questions asked in all runs: 98. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
