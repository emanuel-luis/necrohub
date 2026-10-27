# Model card: necrohub encoders (NER and section classification)

Two heads fine-tuned from `pucpr/biobertpt-clin` on the synthetic necrohub corpus: a token
classifier for de-identification and clinical entities, and a sequence classifier for the 8 report
sections. Synthetic data only; not for clinical or forensic use.

## Model details

| | |
| --- | --- |
| Base model | `pucpr/biobertpt-clin`, a BERT-base model pretrained on Portuguese clinical text. Its model card and repository declare no licence. Citation: Schneider et al., 2020, "BioBERTpt: A Portuguese Neural Language Model for Clinical Named Entity Recognition", Proceedings of the 3rd Clinical Natural Language Processing Workshop. |
| `encoder-ner` | Token classification with 35 BIO tags: `O` plus B and I for the 17 entity labels in `contracts/spans.py` (11 PII, 6 clinical). CNPJ and EMAIL never occur in the corpus, so 15 labels are trained and scored. |
| `encoder-sections` | Sequence classification of a section body (header line removed) into 8 labels, PREAMBLE to SIGNATURE. |
| Weights | Not published, because the base model declares no licence. Retraining with the command below gives equivalent, not bit-identical, weights (decision 17). |
| Code | `src/necrohub/train/` and `src/necrohub/evaluate/` at commit `a348251`. |

## Training data

The 1500-case corpus produced with `generate.n_cases=1500` and seed 7. Its datasets sha256 is
`a0170793…7dec6`, the value in `docs/results/full-corpus.json`. Splits are by case: 1020 train,
128 validation, 352 test. Cases written with template `t3` appear only in the test split, so the
`heldout_templates` scope measures the models on report wording they never saw. Each NER record is
one report section with its gold spans (8160 train and 1024 validation records); each section
record is one section body.

## Training procedure

| Setting | Value |
| --- | --- |
| Epochs | 3 (3060 optimizer steps per task) |
| Batch size | 8 |
| Learning rate | 5e-5, linear schedule, warmup 10 % of steps |
| Weight decay | 0.01 |
| Max length / window overlap | 256 tokens / 64 tokens |
| Precision | fp16, gradient checkpointing |
| Seed | 7 |
| Hardware | NVIDIA GeForce GTX 1650 4 GB, Windows 11, driver 591.59 |
| Software | torch 2.14.1+cu126, transformers 5.20.2 |
| Peak GPU memory | 3355 MiB |
| Training time | NER 8468 s (2 h 21 min), sections 8133 s (2 h 16 min) |
| Final validation loss | NER 0.0001, sections 0.0001 |

Checkpoints are saved every 200 steps under `models/.checkpoints/`; rerunning the same command
resumes from the last one. The first attempt of this run stopped when the sections training
process failed to start (Windows status 0xC0000142); rerunning resumed at that stage.

## Evaluation

Test split only, in two scopes. `random` holds 127 test cases from templates seen in training;
`heldout_templates` holds the 225 `t3` cases. Strict matching needs the same label and the same
character boundaries. PII character recall is the share of gold PII characters covered by any
predicted PII span, whatever its label; for de-identification it is the number that matters,
because a covered value is redacted. The regex baseline only predicts PII labels, so its scores
are over PII labels.

### De-identification (PII labels)

| Scope | System | Strict P | Strict R | Strict F1 | PII char recall | PII spans missed |
| --- | --- | --- | --- | --- | --- | --- |
| heldout_templates | regex | 1.0000 | 0.5000 | 0.6667 | 0.4716 | 2025 |
| heldout_templates | encoder | 0.9444 | 0.9444 | 0.9444 | 1.0000 | 0 |
| random | regex | 0.9917 | 0.6105 | 0.7557 | 0.5944 | 905 |
| random | encoder | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |

The regex finds no places (LOC) in either scope and no person names (PER) on the held-out
template (PER F1 0.4224 on `random`); those are most of the spans it misses. The encoder misses no PII character on the held-out template.

### Entities by label, held-out template (strict F1)

| Label | Support | Encoder | Regex |
| --- | --- | --- | --- |
| PER | 1125 | 1.0000 | 0.0000 |
| LOC | 675 | 1.0000 | 0.0000 |
| DATE | 675 | 1.0000 | 1.0000 |
| DOC_NUMBER | 450 | 0.8000 | 0.6667 |
| ADDRESS | 225 | 1.0000 | 1.0000 |
| CPF | 225 | 1.0000 | 1.0000 |
| CRM | 225 | 0.0000 | 1.0000 |
| PHONE | 225 | 1.0000 | 1.0000 |
| RG | 225 | 1.0000 | 1.0000 |
| ORGAN | 1598 | 1.0000 | n/a |
| FINDING | 461 | 1.0000 | n/a |
| INJURY | 242 | 1.0000 | n/a |
| CAUSE_OF_DEATH | 225 | 1.0000 | n/a |
| MANNER_OF_DEATH | 225 | 1.0000 | n/a |
| SUBSTANCE | 211 | 1.0000 | n/a |

All labels, held-out template: micro F1 0.9679, macro F1 0.9200. On the `random` scope every label
scores 1.0000.

The one systematic error: on the held-out template all 225 CRM numbers are labelled DOC_NUMBER
(225 CRM false negatives, 225 DOC_NUMBER false positives). Template `t3` writes the examiner's
registration in a context the model never saw for CRM, and it falls back to the closest number
type. The digits are still redacted, which is why PII character recall stays at 1.0.

### Sections

Accuracy and macro F1 are 1.0000 in both scopes (1016 sections in `random`, 1800 in
`heldout_templates`).

## Limitations

- The corpus comes from three templates per section, so the `random` scope is optimistic. The
  `heldout_templates` scope is the better estimate of how the models handle new wording, and it
  is still synthetic and narrow.
- Names and places come from Faker's pt_BR provider; document numbers are checksum-invalid fakes.
  Nothing here was validated on real reports, and real reports would score lower.
- The label confusion above shows the NER model leans on context words. More template variety,
  or the LLM realizer, is the fix.
- The serving index of this run was still redacted by the regex de-identifier: 4500 of its 12029
  chunks contain a gold PII string (`index.json`). Preprocessing can use this NER model
  (`preprocess.deidentifiers: [regex, hf_ner]` with `preprocess.hf_ner.model` set to a trained
  model directory), but the model comes from `train_ner`, downstream of `preprocess`, so that is a
  second run with an existing model. This run did not measure it.
- The regex baseline covers PII only; the clinical labels have no baseline.

## Reproduce

```bash
make install-gpu        # cu126 torch, a multi-GB download
uv run necrohub gpu-check
uv run dvc exp run -S generate.n_cases=1500 -S train.base_model=pucpr/biobertpt-clin \
  -S train.fp16=true -S train.gradient_checkpointing=true -S train.learning_rate=5e-5 \
  -S train.epochs=3
```

The metrics of this run are in `docs/results/encoder-biobertpt-clin/` (`run.json` has the command,
versions and durations). GPU training is not bit-identical across machines (decision 17); the data
stages are.
