# CE5 model source

This directory contains model architecture code. It is separate from
`ce5/models/`, which contains generated training checkpoints and is ignored by
Git.

Architecture files:

- `independent_latent_heads.py`: the original CE5 model and control baseline.
- `directional_latent_matcher.py`: directional, interacting-latent architecture.
- `registry.py`: reconstructs the correct architecture from checkpoint metadata.

The compatibility module `ce5/model.py` continues to export the original model
so existing commands and checkpoints remain valid.
