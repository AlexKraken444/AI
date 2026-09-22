# Kraken Context 3: training data

The corpus combines original project examples, new authored context exercises, and a filtered Russian subset of **OpenAssistant Conversations Dataset (OASST1)**.

## External source and attribution

- Authors: OpenAssistant contributors / LAION. See Köpf et al., *OpenAssistant Conversations — Democratizing Large Language Model Alignment* (2023), https://arxiv.org/abs/2304.07327.
- Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1
- Source repository: https://github.com/LAION-AI/Open-Assistant
- Declared license: **Apache License 2.0**. License text: https://www.apache.org/licenses/LICENSE-2.0
- A copy of the license is included at `licenses/Apache-2.0.txt`.
- Exact revision, file SHA-256 and generated split hashes are recorded in `data_manifest.json`.
- Modifications: select Russian reviewed messages, remove deleted messages, filter quality and content annotations, reject URL/email/phone-like content, reconstruct intact dialogue chains, retain highly ranked responses, deduplicate identical input contexts, and split by normalized root question. Source message/tree IDs remain in prepared examples.
- Raw and prepared third-party messages stay in ignored `training-data/`; they are not served by the app or uploaded as browser memory. The preprocessing script permits reproducing the corpus from its public source.

## Split policy

Training, validation and test groups are assigned before tokenization. All branches sharing a normalized root question stay in one split. Synthetic variations sharing the same facts stay in one split. Exact duplicate input contexts are removed globally. The tokenizer is fitted only on training text. Validation selects candidate checkpoints; test examples are not used for gradient updates or selection.

Original version-2 examples remain in the corpus to preserve basic skills. Some were seen by the old model; the old/new evaluation excludes that category and instead uses new exercises and OASST1 test conversations.

## Limits

The majority of examples are synthetic and cannot substitute for broad language pretraining. Annotation-based filtering does not guarantee factual correctness or absence of all personal/unsafe content. Only short complete answers that fit the configured training limit are used; long answers are rejected, not silently cut off and labelled as complete. Counts of usable/rejected answers are in checkpoint metrics. These filters bias the corpus toward short answers and a narrow range of tasks.

Generation exact-match checks are appropriate for the synthetic questions with one prescribed answer, not for judging arbitrary conversation. The dialogue metric is negative log likelihood per UTF-8 byte over the same answer prefixes, because token losses from different tokenizers are not directly comparable. It is not a factuality or safety score.

## Reproduce

```bash
python training/prepare_data.py
python training/run.py --steps 3200
python training/evaluate.py --model neural/checkpoint --output training-output/baseline.json
python training/evaluate.py --model training-output/context3 --output training-output/candidate.json
```

Training candidates are written to `training-output/`. The serving checkpoint is not replaced automatically by a training run.

## This run

18,152 prepared train examples; 982 validation; 1,052 test. Length filtering retained 17,880 train examples (240 OASST1, 17,640 authored) and 971 validation examples. The held-out comparison uses 132 new synthetic cases and all 40 OASST1 test examples. The candidate improved those measured tasks but still produces unreliable free-form language; it is published as an explicit opt-in mode. Full scalar results are in `comparison.json`.
