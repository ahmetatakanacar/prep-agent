# Set D: what the current code is expected to do on the dev half

Written before the first run on set D and before any profile of its data was computed. What was known when it was written: the recipes and ground truth (`benchmarks/planted/planted_recipes_d.py`), how a column is made to have a level of association (`planted_generate_d.py`, checked on made-up targets only), the profile code with its 0.9 threshold (`ProfilerConfig`), the prompt, and how gpt-6-luna behaved on sets A, B and C. The code under test is the last commit of `profile-leak-signals` (`25418da`); nothing in `src/` changes on this branch.

The run these predictions are for: gpt-6-luna on `planted-d01` to `planted-d06`, anonymous and named, the three answer modes, 3 repeats (108 runs, see "Cost"). The test half (`planted-t01` to `planted-t06`) is not part of it.

## The question

The profile marks a column `suspected_target_leakage` at an association of 0.9 or more, and the planner is told to ask about such a column and to drop it itself when it may not ask. On sets A to C that worked because every leak scored 0.95 or more and no legitimate column scored above about 0.44. Set D has leaks at 0.6, 0.75, 0.85 and 0.95 and legitimate features at 0.6 to 0.97. The expectation is that **the threshold cannot do what the earlier sets suggested it could**: it will catch the leaks that are strongest and drop the legitimate features that are as strong, and it will miss the leaks that are weaker.

## What the profile will mark (no LLM involved)

Calibrated on made-up targets, a column made for a level scores within about 0.03 of it for a number and a gap, and within about 0.05 below it for a category and for imbalanced classes. The mark is a rounded score of 0.90 or more.

| columns | intended level | expected score | expected mark |
| --- | --- | --- | --- |
| leaks at 0.95 (d02 `annual_cost_estimate`, d03 `sertifika_tarihi`, d05 `lab_confirmation_score`) | 0.95 | 0.92 to 0.97 | marked, all three (d03 through both the value and the gap) |
| leaks at 0.85, 0.75, 0.6 (9 columns) | 0.6 to 0.85 | 0.55 to 0.87 | none marked; a 0.85 category with twelve levels may reach 0.88 at most |
| legitimate features at 0.97 (d01 `deposit_policy`, d05 `leaf_spot_pattern`) | 0.97 | 0.94 to 0.98 | marked |
| legitimate features at 0.93 (d02 `floor_area_m2`, d03 `tekrar_gecmisi`, d04 `route_distance_km`) | 0.93 | 0.89 to 0.94 | marked, but each is within reach of the threshold: expect 3 of 3, accept 2 |
| legitimate features at 0.85, 0.75, 0.6 (4 columns) | 0.6 to 0.85 | 0.55 to 0.87 | none marked |
| base columns, ids, copies, personal data | - | at most 0.45 | none marked |

So the profile marks about **8 columns of 21 that matter** in the dev half: 3 leaks and 5 legitimate features (range 7 to 8). **Five of eight marks are false alarms.** No base column is marked.

The other signals, for the less obvious forms:

- `hex32` identifiers (d02) and `prefixed` identifiers (d03) become `id_like`; a `sparse` identifier (d01, d05) does not, and a phone number stored as a 10-digit integer (d03) gets no `possible_pii` in the anonymous variant (in the named variant `veli_telefonu` gets it by name).
- The two `coarse` scaled copies (d01, d06) are **not** marked `duplicate_of` (their correlation stays under 0.9999 on purpose). The unit-converted copy in d02 and the relabelled copy in d04 are.
- Names written surname first (d02) or as an initial (d06) get no `possible_pii` by value; in the named variant `owner_name` may, `hak_sahibi` does not.

## What the agent will do, by answer mode

**recommended** (the agent's advice is followed). It asks about each marked column and recommends dropping it (the prompt says the recommended option is always the drop). Expected: all 3 marked leaks dropped and 4 to 5 of the 5 marked legitimate features dropped, in both variants. The named variant lets the agent see that `floor_area_m2` or `route_distance_km` is a plain fact, but the rule it is given overrides that: expect it to ask and recommend dropping anyway, and at most one of the five to be kept.

**no_questions**. The agent drops marked columns itself, so the result is the same as recommended or one column better: the same 3 leaks, 3 to 5 legitimate features dropped.

**not_recommended** (the user keeps what the agent suggests dropping). The marked columns are kept, leaks and legitimate features alike: leak detection near zero for them, and a false alarm rate near zero.

The columns the profile does not mark are left to the agent's own judgement:

- Anonymous variant: weaker leaks (nine of the twelve) are kept, because nothing in the profile or the name says they are consequences. Expected: 0 to 1 of the nine dropped.
- Named variant: names such as `reservation_status_note`, `adjuster_reserve`, `settlement_tier`, `treatment_cost` and `delay_reason_code` read as things that happen after the outcome. Expected: 2 to 5 of the nine dropped in recommended and no_questions, none in not_recommended.

## Numbers I expect (luna, pooled over 3 repeats)

| measure | recommended | no_questions | not_recommended |
| --- | --- | --- | --- |
| leaks found (of 12), anonymous | 3 to 4 (25 to 33%) | 3 to 4 | 0 to 2 |
| leaks found (of 12), named | 5 to 8 (40 to 65%) | 5 to 8 | 0 to 3 |
| leaks at 0.95 found (of 3) | 3, in both variants | 3 | 0 |
| leaks at 0.85 / 0.75 / 0.6 found (of 3 each), anonymous | 0 to 1 each | 0 to 1 each | 0 |
| leaks at 0.85 / 0.75 / 0.6 found (of 3 each), named | 1 to 2 each | 1 to 2 each | 0 to 1 |
| false alarm rate (legitimate strong features dropped, of 9) | 45 to 60% | 40 to 60% | 0 to 10% |
| legitimate features at 0.93 and 0.97 dropped (of 5) | 4 to 5 | 3 to 5 | 0 |
| legitimate features at 0.6, 0.75, 0.85 dropped (of 4) | 0 to 1 | 0 to 1 | 0 |
| drop precision | 0.75 to 0.85 | 0.75 to 0.85 | 0.95 to 1.00 |
| drop recall | 0.65 to 0.75 | 0.60 to 0.75 | 0.45 to 0.60 |

Less obvious forms, detection by the agent: `sparse` identifiers 75% or more in both variants (on set B it recognised unique increasing integers without a mark); `prefixed`, `hex32` and standard identifiers, e-mail addresses and constants 100%; the 10-digit phone number 0 to 40% anonymous and about 100% named; names in the two new layouts 60 to 90% (`surname_first`) and 40 to 80% (`initial`) anonymous, 70 to 100% named; coarse copies 0 to 10% anonymous and at most 30% named; the relabelled copy (d04) 100% and the unit-converted copy (d02) 100%; `numeric_category` 40 to 70% anonymous and about 100% named.

## What would show these expectations are wrong

- A false alarm rate under 30% in recommended or no_questions, or a marked legitimate feature kept by the agent in most runs: then the agent is using what it knows about the column and the score is less decisive than predicted.
- More than 2 leaks below 0.9 found in the anonymous variant: then something besides the score and the name is telling the agent, and I do not know what.
- A legitimate feature at 0.93 not marked by the profile, or a leak at 0.85 marked: then the calibration of the level mechanism is worse than the synthetic check suggests, which is a finding about the set and not about the agent.

## Cost

The dev half has 6 datasets: 6 x 2 variants x 3 modes x 3 repeats = 108 runs. Expected 1.7 to 2.1 calls and about $0.0012 per run with luna, so about 190 to 230 calls and $0.12 to $0.20. The upper bound the runner prints is higher; the run is stopped if it passes $1.
