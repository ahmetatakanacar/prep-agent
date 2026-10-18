# Set D test half, set C and the OpenML benchmark: what the final code is expected to do

Written before any D-test data was generated, profiled or run, and before the closing runs on set C and OpenML. What was known when it was written: the recipes and ground truth of the test half (`planted_recipes_d.py`, the truth files), how a column is made to have a level of association (checked on made-up targets), the profile code and its thresholds, the prompt, and the dev results of the first and second round of fixes (`docs/eval-d-round2.md`). Nothing in the test data was looked at. The code is the last commit of `eval-set-d` before the closing runs (`1402ac1` plus documents); it does not change from here on.

The runs: gpt-6-luna, anonymous and named, the four answer modes (recommended, not_recommended, no_questions, oracle), 3 repeats, on `planted-t01` to `planted-t06` (144 runs), on set C (`planted-c01` to `planted-c08`, 192 runs) and the OpenML benchmark (27 agent runs).

## What the test half contains

Nine legitimate strong features (levels 0.6, 0.75, 0.85 twice, 0.93 three times, 0.97 twice) and twelve leaks of partial strength (0.6, 0.75, 0.85, 0.95, three of each), in other domains than the dev half: phone resale, ride cancellation, stock-outs, solar yield, library returns, restaurant revenue. Gap leaks at 0.85 (`surucu_iptal_nedeni`) and 0.95 (`ceza_odendi_tarihi`); the old fault types in the less obvious forms.

## What the profile will mark (no LLM)

The levels were calibrated on made-up targets and reproduced on the dev half within 0.03 for numbers and gaps and a little more below for categories and imbalanced classes. For the test half I expect the same:

- **Marked at 0.9 or more** (suspected): the three leaks at 0.95 (`sold_price_bracket`, `emergency_order_qty`, `ceza_odendi_tarihi`) and the five legitimate features at 0.93 and 0.97 (`launch_price`, `days_of_cover`, `gunes_isinimi`, `alim_noktasi_tipi`, `odunc_sinifi`). The 0.93 ones are within reach of the threshold: expect 2 or 3 of the 3, as in the dev half where two scored 0.92. So 7 to 8 columns are marked suspected, and 4 to 5 of them are legitimate.
- **Marked at 0.6 to 0.9** (high): the leaks at 0.75 and 0.85 (`buyer_negotiated_discount`, `restock_priority`, `hatirlatma_epostasi_sayisi`, `surucu_iptal_nedeni`, `performans_sinifi`, `delivery_commission_total`), and the legitimate features at 0.75 and 0.85 (`seat_count`, `tahmini_varis_dk`, `yenileme_sayisi`). The leaks and legitimate features at level 0.6 are on the edge: expect the leaks (`destek_bileti_suresi`, `alarm_sayisi`, `revenue_tier_label`) to score 0.52 to 0.60 and at most one to be marked; the legitimate 0.6 feature (`reorder_gap`) to be unmarked. A twelve-level category of a number scores a few hundredths under its level.
- No base column, identifier or copy is marked (the largest score among them was 0.42 in the dev half).
- The relabelled copies (`colour_label`, `bolge_adi`) and the converted copies with the plain conversion (`panel_yasi_ay`, `avg_ticket_eur`) are `duplicate_of`; the `hex32` identifiers, the prefixed identifier `talep_no` and the dense `store_ref` are `id_like`; the sparse `tesis_no` is not; the 10-digit phone number gets no `possible_pii` in the anonymous variant.

## What the agent will do

The second round made the default of an unclear marked column "drop", with or without questions, and a name that plainly says "known before the outcome" the way to keep one. In the dev half that gave, per mode (I expect the same shape on the test half):

| measure | recommended | no_questions | not_recommended | oracle |
| --- | --- | --- | --- | --- |
| false alarms (of 27), anonymous | 21 to 24 | 21 to 24 | 0 to 3 | 0 to 1 |
| false alarms (of 27), named | 4 to 9 | 4 to 9 | 0 to 3 | 0 to 2 |
| leaks found at 0.75, 0.85 and 0.95, anonymous (of 27) | 25 to 27 | 25 to 27 | 0 to 8 | 25 to 27 |
| leaks found at 0.6, anonymous (of 9) | 0 to 3 | 0 to 3 | 0 to 3 | 0 to 3 |
| plan accuracy, anonymous | 0.84 to 0.92 | 0.84 to 0.92 | 0.80 to 0.90 | 0.95 to 0.99 |
| plan accuracy, named | 0.92 to 0.98 | 0.92 to 0.99 | 0.88 to 0.96 | 0.97 to 1.00 |
| drop precision, anonymous | 0.72 to 0.82 | 0.72 to 0.82 | 0.90 to 1.00 | 0.97 to 1.00 |

- **Oracle plan accuracy**, the one number I expect to move least: anonymous 0.95 to 0.99, named 0.97 to 1.00, because the agent asks about every marked column in the anonymous variant (expect 100% of marked columns asked, both tiers) and an informed user then answers right.
- **Which legitimate features go**: in the anonymous variant every marked one (about 24 of 27 occurrences, the unmarked 0.6 one stays). In the named variant the ones whose names read as plain facts are kept with a reason (`launch_price`, `gunes_isinimi`, `seat_count`, `days_of_cover`); the ones that read as something that happens in the process are dropped (`tahmini_varis_dk`, `alim_noktasi_tipi`, `odunc_sinifi`, `yenileme_sayisi`). That is a guess about names, and the least certain line in this document.
- **Questions** in the anonymous variant: 5 to 7 per run; named: 1.3 to 2.2.
- **Old fault types**: identifiers (sparse, prefixed, hex32, dense), e-mail, names in both layouts, converted and relabelled copies, constants: 100%. The 10-digit phone number: 100% in recommended, no_questions and oracle, 0 to 3 of 3 in not_recommended. `numeric_category` (`grade_code`, `kategori_kodu`): anonymous 0/6 in recommended and no_questions, 6/6 in oracle; named 6/6. Coarse copies are not in the test half.
- **Cost**: 140 to 160 calls per 100 runs at about $0.0017 a run, so the D-test run costs $0.22 to $0.30. No failed runs.

## Set C (the first three sets are the same eight recipes, with leaks at 0.95 and above and weak legitimate columns)

With the final code, in the modes where the planner decides (recommended, no_questions): every column marked 0.9 or more is asked about in the anonymous variant and dropped when it is a leak, which on set C is almost all of them, so I expect leak detection at or above the earlier closing run (anonymous 24/30, named 27/30 to 30/30 on set C) and **no legitimate strong features to lose** (set C has none). The leaks that stay: the 4-level categorical leaks of the regression recipes (scores 0.74 to 0.80, unmarked), found in the named variant by name only. Duplicates, identifiers, personal data and constants as before (100%); `numeric_category` anonymous 0 to 6 of 12, named 12/12. Oracle: plan accuracy 0.95 to 1.00 in both variants. Questions per run in the anonymous variant will rise a little (the lower tier is asked about too).

## OpenML (27 agent runs)

The final code asks about the marked columns of the nine datasets: `boat` (titanic), `OverallQual` (house-prices) and in employee-salaries `2016_gross_pay_received` and `employee_position_title`. Expected: `boat` and `2016_gross_pay_received` dropped in all runs (leak kept 0 of 6 and 0 of 3); `OverallQual` kept (its name says a property of the house) with its reason in the split step; `employee_position_title` dropped in 1 to 3 of 3 runs, which costs employee-salaries about 0.05 to 0.15 of R² against the reference plan, so the dataset will count as a loss under the new noise rule; `underfilled_job_title` kept in 3 of 3. The other seven datasets: scores within noise of the reference (as in every earlier run); valid plans 27 of 27; questions per run 0.9 to 1.5; cost per run $0.0012 to $0.0020.

## What would show these expectations are wrong

- Oracle plan accuracy under 0.93 in the anonymous variant or under 0.95 in the named one on D-test: then the dev half overstated how well the agent asks.
- A false alarm rate in the anonymous variant under 15/27 in recommended or no_questions: then the agent keeps marked columns it cannot tell about, against its rule.
- More than 2 of the 9 leaks at 0.75 and 0.85 missed in the anonymous oracle runs: then the questions are not reaching the columns they should.
- Any failed run, or a set C leak miss among the 0.95-and-above leaks.
