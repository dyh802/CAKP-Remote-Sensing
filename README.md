# CAKP Remote Sensing Prompt Learning Code

This repository contains the code used for the paper:

**CAKP: Confusion-Aware Knowledge Prompting for Few-Shot Remote Sensing Scene Classification**

The implementation is built on the official PromptSRC codebase:

https://github.com/muzairkhattak/PromptSRC

CAKP is implemented as a lightweight, plug-and-play knowledge-guided branch for existing prompt learning backbones. The main modifications are in the `trainers/` directory.

## Main Changes

The following trainer files contain CAKP support:

- `trainers/coop.py`
- `trainers/cocoop.py`
- `trainers/maple.py`
- `trainers/promptsrc.py`

The added branch loads a class-wise knowledge prompt library, encodes it with the frozen CLIP text encoder, and combines the original prompt-learning logits with the knowledge-guided logits:

```text
z = (1 - alpha) * z_original + alpha * z_knowledge
```

The original PromptSRC-style code structure is preserved. The original backbone implementations are kept under:

```text
trainers/original_backbones/
```

Reference CAKP variants used during experiments are kept under:

```text
trainers/cakp_variants/
```

## Enabling CAKP

CAKP is controlled by the following configuration options:

```text
TRAINER.<BACKBONE>.USE_CAC True
TRAINER.<BACKBONE>.CAC_ALPHA 0.5
TRAINER.<BACKBONE>.CAC_ATTR_PROMPT_PATH path/to/knowledge_prompts.json
```

where `<BACKBONE>` can be:

```text
COOP
COCOOP
MAPLE
PROMPTSRC
```

Example:

```bash
python train.py \
  --trainer CoOp \
  --dataset-config-file configs/datasets/UCM.yaml \
  --config-file configs/trainers/CoOp/vit_b16.yaml \
  TRAINER.COOP.USE_CAC True \
  TRAINER.COOP.CAC_ALPHA 0.5 \
  TRAINER.COOP.CAC_ATTR_PROMPT_PATH data/rs_knowledge/attribute_prompts/UCM_coop.json
```

## Datasets

This repository does not redistribute the original remote sensing images. Please download the public datasets from their official or commonly used public sources and arrange them under your local data directory following the dataset configuration used in your experiments. Dataset wrappers and configuration files for the remote sensing datasets are included under `datasets/` and `configs/datasets/`.

The experiments use the following dataset names:

| Dataset name used in this project | Public source |
| --- | --- |
| UCM / UC Merced Land Use | http://weegee.vision.ucmerced.edu/datasets/landuse.html |
| AID | https://captain-whu.github.io/AID/ |
| WHU-RS19 | https://captain-whu.github.io/BED4RS/ |
| NWPU-RESISC45 | https://gcheng-nwpu.github.io/ |
| PatternNetV2 | https://drive.google.com/file/d/1K-GZ2KjQ3hn17JJBrxnmXsTxAFeg2XUT/view?usp=sharing |
| RESISC45v2 | https://drive.google.com/file/d/1Zfsko5swyQqu5HiuRwZe5jIGoUKfBgxq/view?usp=sharing |
| RSICDv2 | https://drive.google.com/file/d/1uhlTHQCHkE0KD04YGBAKsxPgG14eQez_/view?usp=sharing |
| MLRSNetV2 | https://drive.google.com/file/d/1OJrAwU1i9hYe7kEsHIIq_TodJDiwnnAz/view?usp=sharing |

Note: The four Version-2 datasets (`PatternNetV2`, `RSICDv2`, `RESISC45v2`, and `MLRSNetV2`) follow the released domain-generalization datasets from the official APPLeNet repository: https://github.com/mainaksingha01/APPLeNet.

## Generated Data Included in This Repository

The generated knowledge prompt libraries used by CAKP are provided under:

```text
attribute_prompts/
```

The LLM instruction templates used to construct these prompts are provided in:

```text
llm_instruction_templates.txt
```

## Data and Materials License

The generated knowledge prompt files, LLM instruction templates, and related
research materials in this repository are made available under the Creative
Commons Attribution 4.0 International License (CC BY 4.0). See
`LICENSE-DATA.md` for details.

The source code is provided for research and reproducibility purposes. Parts of
the code are adapted from the official PromptSRC repository and follow the
licence terms of the original project.

## Knowledge Prompt Files

The knowledge prompt files are JSON dictionaries. Each key is a class name and each value is the generated textual knowledge prompt for that class. These prompts may include semantic attributes and confusion-aware discriminative cues.

Example:

```json
{
  "runway": "A remote sensing runway scene usually contains long linear paved strips, regular markings, open surrounding areas, and differs from airplane scenes by the absence of dominant aircraft bodies.",
  "harbor": "A harbor scene usually contains docks, ships, water boundaries, and man-made shoreline structures."
}
```

## Notes

- The pretrained CLIP image and text encoders remain frozen.
- The LLM is only used offline to construct the knowledge prompt library.
- No LLM query is required during training or inference.
- The repository preserves the original PromptSRC project structure for compatibility.

## Citation

If you use this code, please cite the CAKP paper and the original PromptSRC repository.



