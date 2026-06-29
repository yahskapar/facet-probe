# Screened ODI Fit Audit

Date: 2026-06-23

## Inputs

- Modal fit: `EMNLP_2026/MMIOS/results/experiments/irt_v6_screened5_modal_2026-06-23_04-42-33_2026-06-23_04-42-34/`
- Correct fit: `EMNLP_2026/MMIOS/results/experiments/irt_v6_screened5_correct_2026-06-23_04-42-33_2026-06-23_04-42-37/`
- Screen: `EMNLP_2026/MMIOS/data/screens/imageset_position_reference_screen.json`

Both fits used the D1 image-set screen: drop `classification == exclude`, keep
`keep` + `borderline`. The fit metadata reports Mantis-Eval 52/70 excluded and
MedFrameQA 5/200 excluded, leaving 213 image-set items in the ODI input.

## Verdict

The screened modal ODI fit is clean and is suitable for Table 2. The screened
correct-outcome fit is usable for `theta_correct` and capability correlations,
but not clean as a whole: its global warning is localized to dataset-difficulty
`mu_beta` hypermeans, while `theta`, `sigma_delta`, `mu_log_sigma`, and
`tau_log_sigma` are well mixed.

## Modal Fit

- Diagnostics: `Rhat_max=1.000`, `ESS_bulk_min=898`, `ESS_tail_min=1597`, 0 divergences.
- Items: `n_items=3612`.
- Table 2 values, ratios against `option_order`:

| Facet | n_items | sigma_pi median | sigma ratio | abs(delta) mean | delta ratio |
|---|---:|---:|---:|---:|---:|
| option_order | 878 | 0.087536 | 1.000 | 0.002619 | 1.000 |
| document_rank_order | 1138 | 0.096574 | 1.103 | 0.009976 | 3.808 |
| evidence_chunk_order | 786 | 0.105018 | 1.200 | 0.011779 | 4.497 |
| image_set_order | 213 | 0.151824 | 1.734 | 0.021610 | 8.250 |
| mixed_modality_order | 597 | 0.264901 | 3.026 | 4.795820 | n/a |

Interpretation: mixed-modality remains largest overall on `sigma_pi`; among the
four MCQ-style facets, screened image-set remains highest on both `sigma_pi` and
`abs(delta)`. The image-set row should still be framed as MedFrameQA-anchored
because screened Mantis contributes only 18 items.

## Correct Fit

- Global diagnostics: `Rhat_max=1.15`, `ESS_bulk_min=19`, `ESS_tail_min=100`, 0 divergences.
- Localized diagnostic check:
  - `mu_beta`: `Rhat_max=1.15`, `ESS_bulk_min=19`, 10/11 parameters above `Rhat > 1.01`.
  - `theta`: `Rhat_max=1.000`, `ESS_bulk_min=1468`, no `Rhat > 1.01`.
  - `sigma_delta`: `Rhat_max=1.000`, `ESS_bulk_min=830`, no `Rhat > 1.01`.
  - `mu_log_sigma`: `Rhat_max=1.000`, `ESS_bulk_min=1502`, no `Rhat > 1.01`.
  - `tau_log_sigma`: `Rhat_max=1.000`, `ESS_bulk_min=589`, no `Rhat > 1.01`.

Use the correct fit for descriptive `theta_correct` ranking and correlations, but
do not describe the whole correct fit as clean.

## Capability Correlations

Using `derived/screened_panel_values.json` 5-facet means and the screened
correct-outcome `theta`:

- Full 18-model panel: Spearman `rho=-0.946`.
- Frontier six-model cluster: Spearman `rho=-0.886`.
- Best-of-family seven-model panel: Spearman `rho=-0.893`.

The existing `rho approx -0.95` headline remains valid. The within-frontier text
should use `rho approx -0.89`, not the earlier provisional `rho approx -0.77`.

## KS Separability

Pairwise KS tests on posterior-mean `sigma_pi` item estimates from the screened
modal fit separate all 10 facet pairs (`p < 0.001`; many p-values underflow to
0). Because the current model strongly shrinks item-level `sigma_pi` within
facet, report this as facet-scale separability, not as a content-level item
classifier.

## Public Release Note

Internal manuscript-edit task lists were omitted from this public note.
