# Training data attribution

Kraken Context 3 was initialized and trained from scratch in this project.
Training includes a filtered Russian subset of OpenAssistant Conversations (OASST1), by OpenAssistant contributors / LAION, under Apache License 2.0.

- Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1
- Revision: fdf72ae0827c1cda404aff25b6603abec9e3399b
- Paper: Köpf et al., OpenAssistant Conversations — Democratizing Large Language Model Alignment (2023), https://arxiv.org/abs/2304.07327
- License copy: `training/licenses/Apache-2.0.txt` in this repository.
- Filtering, original synthetic additions, data splits and limitations: `training/DATA_CARD.md`.

No pretrained language-model weights or private user conversations were used. This notice describes dataset attribution, not an endorsement by OpenAssistant.
