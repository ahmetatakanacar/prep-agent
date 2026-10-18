# Planted-fault evaluation

Started 2026-10-18 08:13 UTC, finished 08:55 UTC.
Models: gpt-6-luna. Variants: anonymous, named. Modes: recommended, not_recommended, no_questions. Repeats: 3. Datasets: 8. Runs recorded: 144.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | must_keep dropped | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 24 (24) | 0.82 (0.80-0.85) | 1.00 | 0.86 | 0 (0.00/run) | 6 (0.25/run) | 2.2 | 2.3 | 0.0017 | 0.0402 |
| gpt-6-luna | named | recommended | 24 (24) | 0.97 (0.97-0.97) | 1.00 | 0.97 | 0 (0.00/run) | 0 (0.00/run) | 1.1 | 2.0 | 0.0013 | 0.0305 |
| gpt-6-luna | anonymous | not_recommended | 24 (24) | 0.54 (0.53-0.55) | 1.00 | 0.55 | 0 (0.00/run) | 0 (0.00/run) | 2.0 | 2.2 | 0.0015 | 0.0370 |
| gpt-6-luna | named | not_recommended | 24 (24) | 0.75 (0.75-0.75) | 1.00 | 0.72 | 0 (0.00/run) | 0 (0.00/run) | 1.1 | 2.0 | 0.0012 | 0.0285 |
| gpt-6-luna | anonymous | no_questions | 24 (24) | 0.80 (0.78-0.82) | 1.00 | 0.82 | 0 (0.00/run) | 6 (0.25/run) | 0.0 | 1.0 | 0.0009 | 0.0209 |
| gpt-6-luna | named | no_questions | 24 (24) | 0.97 (0.95-0.97) | 1.00 | 0.96 | 0 (0.00/run) | 0 (0.00/run) | 0.0 | 1.0 | 0.0007 | 0.0175 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 100% (9/9) | 100% (9/9) | 50% (6/12) | 92% (11/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 33% (4/12) | 100% (6/6) | 100% (9/9) | 50% (6/12) | 100% (12/12) | not scored |
| gpt-6-luna | named | recommended | 100% (9/9) | 100% (9/9) | 75% (9/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 100% (12/12) | 100% (12/12) | not scored |
| gpt-6-luna | anonymous | not_recommended | 0% (0/9) | 0% (0/9) | 0% (0/12) | 42% (5/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 0% (0/12) | 100% (6/6) | 100% (9/9) | 50% (6/12) | 100% (12/12) | not scored |
| gpt-6-luna | named | not_recommended | 0% (0/9) | 0% (0/9) | 0% (0/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 100% (12/12) | 100% (12/12) | not scored |
| gpt-6-luna | anonymous | no_questions | 100% (9/9) | 100% (9/9) | 50% (6/12) | 83% (10/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 8% (1/12) | 100% (6/6) | 100% (9/9) | 58% (7/12) | 100% (12/12) | not scored |
| gpt-6-luna | named | no_questions | 100% (9/9) | 100% (9/9) | 67% (8/12) | 100% (12/12) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (12/12) | 100% (6/6) | 100% (9/9) | 100% (12/12) | 100% (12/12) | not scored |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | recommended | 0.82 | 0.97 | +15 pp | 1.00 | 1.00 | +0 pp |
| gpt-6-luna | not_recommended | 0.54 | 0.75 | +21 pp | 1.00 | 1.00 | +0 pp |
| gpt-6-luna | no_questions | 0.80 | 0.97 | +17 pp | 1.00 | 1.00 | +0 pp |

## Notes

- Questions asked in all runs: 156. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
