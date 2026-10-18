# Planted-fault evaluation

Started 2026-10-18 02:48 UTC, finished 03:03 UTC.
Models: gpt-6-luna. Variants: anonymous, named. Modes: recommended, not_recommended, no_questions. Repeats: 1. Datasets: 8. Runs recorded: 48.

Rates are over the runs that ended with a valid plan; scores of several runs are added up before the ratio is taken. A plan the agent could not make valid has no score and is listed at the end.

## Results by model, variant and mode

| model | variant | mode | runs (ok) | detection | drop precision | drop recall | must_keep dropped | avoided_op violations | questions/run | calls/run | cost/run (USD) | cost total (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 8 (8) | 0.57 | 1.00 | 0.58 | 0 (0.00/run) | 2 (0.25/run) | 1.6 | 2.0 | 0.0018 | 0.0142 |
| gpt-6-luna | named | recommended | 8 (8) | 0.72 | 0.96 | 0.69 | 1 (0.12/run) | 0 (0.00/run) | 0.6 | 1.5 | 0.0012 | 0.0100 |
| gpt-6-luna | anonymous | not_recommended | 8 (8) | 0.47 | 1.00 | 0.44 | 0 (0.00/run) | 0 (0.00/run) | 1.4 | 2.0 | 0.0014 | 0.0111 |
| gpt-6-luna | named | not_recommended | 8 (8) | 0.62 | 1.00 | 0.58 | 0 (0.00/run) | 0 (0.00/run) | 0.6 | 1.5 | 0.0010 | 0.0079 |
| gpt-6-luna | anonymous | no_questions | 8 (8) | 0.55 | 1.00 | 0.56 | 0 (0.00/run) | 2 (0.25/run) | 0.0 | 1.1 | 0.0011 | 0.0090 |
| gpt-6-luna | named | no_questions | 8 (8) | 0.68 | 0.96 | 0.64 | 1 (0.12/run) | 0 (0.00/run) | 0.0 | 1.0 | 0.0010 | 0.0078 |

## Detection by fault type

Handled columns out of planted columns, over all ok runs of the group.

| model | variant | mode | leak_linear | leak_nonlinear | leak_categorical | id_sequential | id_uuid | pii_email | pii_phone | pii_name | duplicate_exact | duplicate_scaled | numeric_category | constant | sentinel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | anonymous | recommended | 0% (0/3) | 0% (0/3) | 0% (0/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 50% (2/4) | 100% (4/4) | not scored |
| gpt-6-luna | named | recommended | 33% (1/3) | 67% (2/3) | 25% (1/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 100% (4/4) | 100% (4/4) | not scored |
| gpt-6-luna | anonymous | not_recommended | 0% (0/3) | 0% (0/3) | 0% (0/4) | 50% (2/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 25% (1/4) | 0% (0/2) | 0% (0/3) | 75% (3/4) | 100% (4/4) | not scored |
| gpt-6-luna | named | not_recommended | 0% (0/3) | 0% (0/3) | 0% (0/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 100% (4/4) | 100% (4/4) | not scored |
| gpt-6-luna | anonymous | no_questions | 0% (0/3) | 0% (0/3) | 0% (0/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 50% (2/4) | 50% (1/2) | 0% (0/3) | 50% (2/4) | 100% (4/4) | not scored |
| gpt-6-luna | named | no_questions | 33% (1/3) | 0% (0/3) | 25% (1/4) | 100% (4/4) | 100% (3/3) | 100% (3/3) | 100% (3/3) | 100% (4/4) | 0% (0/2) | 0% (0/3) | 100% (4/4) | 100% (4/4) | not scored |

## Anonymous against named

The columns, values and faults are the same in both variants; only the names differ.

| model | mode | detection anonymous | detection named | detection difference | precision anonymous | precision named | precision difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-6-luna | recommended | 0.57 | 0.72 | +15 pp | 1.00 | 0.96 | -4 pp |
| gpt-6-luna | not_recommended | 0.47 | 0.62 | +15 pp | 1.00 | 1.00 | +0 pp |
| gpt-6-luna | no_questions | 0.55 | 0.68 | +12 pp | 1.00 | 0.96 | -4 pp |

## Notes

- Questions asked in all runs: 34. Answers with no other option to pick (chosen as the recommended one, marked `forced` in results.json): 0.
- `sentinel` columns are in the data but not scored: no op turns a special value into a missing value.
