# Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models

<p align="center">
Please star this repo if you find it useful and cite our work if you use Facet-Probe in your research.
</p>

<p align="center">
Paper: TBD | arXiv: TBD
</p>

The official repository for the paper "Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models" and its corresponding Facet-Probe evaluation framework.

Facet-Probe audits whether multimodal large language models are robust to order-irrelevant input permutations across option, evidence-chunk, document-rank, image-set, and mixed-modality ordering facets.

## Release Status

This repository is currently a release scaffold. We plan to release the full evaluation code and related artifacts soon, including prompt templates, permutation indices, aggregation scripts, and reproducibility documentation.

We do not plan to redistribute upstream dataset content. Dataset-backed evaluations will load from the original sources, subject to the corresponding dataset licenses and terms of use.

## Setup

Setup instructions will be added with the full code release.

## Usage

The planned release will include:

- Facet-Probe loaders and adapters for the datasets and models reported in the paper.
- Prompt templates and facet-specific permutation grammars.
- Permutation indices, model-output artifacts where redistributable, and aggregation scripts.
- Reproduction instructions for the main tables and figures.
- Templates for adding new datasets, ordering facets, and model adapters.

## Citation

If you find our paper or this repository useful for your research, please cite our work.

```bibtex
@misc{paruchuri2026same,
  title = {Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models},
  author = {Paruchuri, Akshay and Koyejo, Sanmi and Adeli, Ehsan},
  year = {2026},
  archivePrefix = {arXiv},
  eprint = {TBD},
  note = {TBD}
}
```

## License

```text
Apache License 2.0
```

The code in this repository is released under the Apache License 2.0. Dataset content, model outputs, and third-party dependencies may be subject to their own licenses or terms; the full release will document those constraints alongside the relevant artifacts.
