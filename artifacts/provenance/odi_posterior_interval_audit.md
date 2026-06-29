# ODI posterior interval audit

Date: 2026-06-24

Paper target: arXiv appendix ODI methodology section.

Source artifact:
`MMIOS/results/experiments/irt_v6_screened5_modal_2026-06-23_04-42-33_2026-06-23_04-42-34/irt_v4_modal_idata.nc`

Purpose: support the arXiv appendix statement that ODI posterior
uncertainty is reported off-figure, rather than adding error bars to
the dense descriptive figures.

## Computation

- Opened the `posterior` group from the screened v6 modal-outcome
  NetCDF artifact with `xarray` / `h5netcdf`.
- Used 4 chains x 1500 posterior draws = 6000 posterior draws.
- For `sigma_pi`, computed the draw-wise median of `sigma_item` within
  each facet.
- For systematic offsets, reported the direct posterior hyper-scale
  `sigma_delta` within each facet.
- Used the shortest 89% highest-density interval over posterior draws.
- Separately verified that the Table 2 point summaries are reproduced
  by the script definition: median of per-item posterior-mean
  `sigma_item`, and mean absolute posterior-mean `delta` over
  `(facet, dataset, ordering)` cells.

## Values added to appendix

| Facet | `sigma_pi` draw-wise median [89% HDI] | `sigma_delta` median [89% HDI] |
|---|---:|---:|
| option_order | 0.086091 [0.050446, 0.118783] | 0.012044 [0.000000, 0.033048] |
| document_rank_order | 0.093481 [0.053761, 0.136924] | 0.027487 [0.000003, 0.062920] |
| evidence_chunk_order | 0.102758 [0.057957, 0.146401] | 0.027626 [0.000056, 0.059650] |
| image_set_order | 0.147333 [0.077929, 0.224815] | 0.063795 [0.000011, 0.143136] |
| mixed_modality_order | 0.246045 [0.094650, 0.415704] | 2.363502 [1.972798, 2.764298] |

## Table 2 replication check

| Facet | `sigma_pi` median of item posterior means | mean absolute posterior-mean `delta` | `delta_n` |
|---|---:|---:|---:|
| option_order | 0.087535722 | 0.002619391 | 12 |
| document_rank_order | 0.096573543 | 0.009975625 | 12 |
| evidence_chunk_order | 0.105017791 | 0.011778594 | 12 |
| image_set_order | 0.151824298 | 0.021609771 | 12 |
| mixed_modality_order | 0.264901234 | 4.795819522 | 18 |

## Interpretation note

The appendix interval table is deliberately not the same object as
Table 2's `|delta|` point summary. Table 2 follows the production
summary script and reports mean absolute posterior-mean
`delta_{f,d,o}`. The appendix reports `sigma_delta`, the model's direct
posterior hyper-scale for systematic permutation offsets. This avoids
presenting a noisy absolute-value posterior functional as if it were
the Table 2 ratio interval.
