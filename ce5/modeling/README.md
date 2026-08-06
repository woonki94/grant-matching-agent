# CE5 model source

This directory contains model architecture code. It is separate from
`ce5/models/`, which contains generated training checkpoints and is ignored by
Git.

Architecture files:

- `independent_latent_heads.py`: the original CE5 model and control baseline.
- `independent_pair_aware_heads.py`: independent-v2 experts with separate
  target/candidate attention, explicit pair interactions, and private scorers.
- `logit_aware_router.py`: a zero-initialized, feature-aware router extension
  that can be trained on top of an independent-v2 checkpoint.
- `directional_latent_matcher.py`: directional, interacting-latent architecture.
- `directional_private_experts.py`: the same directional refinement with a
  separate output scorer for each latent.
- `registry.py`: reconstructs the correct architecture from checkpoint metadata.

The compatibility module `ce5/model.py` continues to export the original model
so existing commands and checkpoints remain valid.
