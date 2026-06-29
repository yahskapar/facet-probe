# CTA and Think-Budget Summary Provenance

These compact public summaries back the Figure 4 CTA and think-budget claims
without redistributing upstream dataset content or raw provider outputs.

The raw JSONL provider outputs used to compute these summaries are not shipped
in `v0.0.1`; the public release includes the compact aggregate CSVs.

Flip-rate rule:

- use `answer_normalized` when present;
- otherwise map answer letters back through `sequence` when the sequence is an
  option permutation;
- otherwise use answer letters directly, which is the evidence-order case for
  MedXpertQA.

`cta_flip_summary.csv` reports baseline-vs-CTA rows, plus CTA+multi-pass rows
for the MedXpertQA anti-synergy claim. `think_budget_sweep.csv` reports the
hard MedXpertQA and easy MMLU-Pro budget sweeps used in the figure.

The paper's uncertainty-clean MedXpertQA subset is discussed in the manuscript,
but the item-level subset labels and raw model-output JSONL are not shipped in
this initial public artifact release.
