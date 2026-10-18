# Planted-fault evaluation

Started 2026-10-18 03:06 UTC, finished 03:22 UTC.
Models: gpt-6.1-sol. Variants: anonymous, named. Modes: recommended, not_recommended, no_questions. Repeats: 1. Datasets: 8. Runs recorded: 48.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | must_keep dropped | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.1-sol | anonymous | recommended | 8 (8) | 0.62 | 1.00 | 0.58 | 0 (0.00/run) | 2 (0.25/run) | 1.1 | 1.6 | 0.0195 | 0.1557 |
| gpt-6.1-sol | named | recommended | 8 (8) | 0.85 | 0.91 | 0.83 | 3 (0.38/run) | 0 (0.00/run) | 1.4 | 1.8 | 0.0210 | 0.1684 |
| gpt-6.1-sol | anonymous | not_recommended | 8 (8) | 0.45 | 1.00 | 0.50 | 0 (0.00/run) | 0 (0.00/run) | 1.1 | 1.6 | 0.0131 | 0.1047 |
| gpt-6.1-sol | named | not_recommended | 8 (8) | 0.62 | 1.00 | 0.58 | 0 (0.00/run) | 0 (0.00/run) | 1.1 | 1.8 | 0.0160 | 0.1279 |
| gpt-6.1-sol | anonymous | no_questions | 8 (8) | 0.62 | 1.00 | 0.58 | 0 (0.00/run) | 2 (0.25/run) | 0.0 | 1.0 | 0.0134 | 0.1073 |
| gpt-6.1-sol | named | no_questions | 8 (8) | 0.78 | 0.96 | 0.75 | 1 (0.12/run) | 0 (0.00/run) | 0.0 | 1.0 | 0.0140 | 0.1121 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.1-sol | anonymous | recommended | 0% (0/3) | 0% (0/3) | 0% (0/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 100% (4/4) | 100% (4/4) | not scored |
| gpt-6.1-sol | named | recommended | 100% (3/3) | 67% (2/3) | 100% (4/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 100% (4/4) | 100% (4/4) | not scored |
| gpt-6.1-sol | anonymous | not_recommended | 0% (0/3) | 0% (0/3) | 0% (0/4) | 50% (2/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 75% (3/4) | 0% (0/2) | 0% (0/3) | 0% (0/4) | 100% (4/4) | not scored |
| gpt-6.1-sol | named | not_recommended | 0% (0/3) | 0% (0/3) | 0% (0/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 100% (4/4) | 100% (4/4) | not scored |
| gpt-6.1-sol | anonymous | no_questions | 0% (0/3) | 0% (0/3) | 0% (0/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 100% (4/4) | 100% (4/4) | not scored |
| gpt-6.1-sol | named | no_questions | 67% (2/3) | 33% (1/3) | 75% (3/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 100% (4/4) | 100% (4/4) | not scored |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6.1-sol | recommended | 0.62 | 0.85 | +22 pp | 1.00 | 0.91 | -9 pp |
| gpt-6.1-sol | not_recommended | 0.45 | 0.62 | +18 pp | 1.00 | 1.00 | +0 pp |
| gpt-6.1-sol | no_questions | 0.62 | 0.78 | +15 pp | 1.00 | 0.96 | -4 pp |

## Notes

- Questions asked in all runs: 38. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
