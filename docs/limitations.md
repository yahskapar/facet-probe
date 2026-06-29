# Limitations

This repo mirrors the paper's limitations and adds release-specific boundaries.

- Same-ordering control is scoped to a Gemini subset, not the full 18-model panel.
- `K=6` is a subsample of all permutations when `n! > 6`; any-flip is K-sensitive.
- Temperature-0 provider APIs can still have backend nondeterminism.
- Clean image-set inference is MedFrameQA-anchored because most Mantis-Eval items are position-referential.
- Mixed-modality sem-flip uses LLM-judge labels and carries measurement variance.
- Closed-source APIs can drift after the May 4-25, 2026 access window.
- This initial public repo ships aggregate and compact derived artifacts, not the full internal raw experiment tree.
