# Screened Decoder Decomposition Audit

Date: 2026-06-23

Source artifact:
`EMNLP_2026/MMIOS/results/decoder_decomp_screened.{json,md,_by_facet.csv}`

Run wrapper:
`EMNLP_2026/arxiv_finalization/scripts/run_screened_decoder_decomp.sh`

## Input Check

- Production-only grids: 70 total = 26 `repgrid_arxiv_2026-06-02_04-19-31` + 44 `repgrid_local_2026-06-02_05-09-59`.
- Excluded by the corrected wrapper: 4 smoke grids and all `*_xv_*` follow-up folders.
- Kept records: 878,700.
- Screen-dropped records: 23,940.
- Screen policy: drop `classification == exclude` for `image_set_order`; keep `keep` and `borderline`.
- Screen counts: Mantis-Eval 52/70 excluded, 18 clean retained; MedFrameQA 5/200 excluded, 195 clean retained.

## D2 Numbers

Gemini option-order consistency delta collapses from T=0 to average T>0:

- Flash: `0.0491 -> 0.0221`.
- Pro: `0.0517 -> 0.0174`.

Gemini option-order decoder-cleaned `acc_swing` is comparatively stable:

- Flash: `0.1729 -> 0.1407`.
- Pro: `0.1339 -> 0.1017`.

Open-weight T>0 `acc_swing` remains much larger on the clean categorical facets:

- `option_order`: local mean `0.4580` vs Gemini mean `0.1212`.
- `evidence_chunk_order`: local mean `0.2908` vs Gemini mean `0.0997`.
- `document_rank_order`: local mean `0.1635` vs Gemini mean `0.0520`.

Image-set caveat:

- Pooled clean image-set is `n=213`, but that is MedFrameQA `n=195` plus Mantis-Eval `n=18`.
- Clean Mantis is too small and flat for Gemini to lead with: Flash `acc_swing=0` at all temperatures; Pro `acc_swing=0.0556` at T=0 and `0` at T>0.
- MedFrameQA drives the clean Gemini image-set effect: at T=0, Pro/Flash floors are `0.0296/0.1003`, cross-ordering rates are `0.1018/0.1364`, and `acc_swing` is `0.1487/0.1744`.

## Paper Implication

D2 remains C:

- Main text: keep the T=0 Gemini same-ordering paragraph, corrected and scoped.
- Appendix: keep the acc_swing-led Gemini T-sweep plus open-weight extension.
- Do not use `decoder_decomp_all.*` for arXiv exact values; it pools 4 smoke grids.
- Do not lead with pooled image-set decomposition; use screened MedFrameQA or omit image-set from the decomposition headline.
