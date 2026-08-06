# CE5 shared training code

This package contains the trainer implementations and their shared components:

- `train_independent_latent_heads.py`: original CE5 latent-head trainer;
- `train_directional_latent_matcher.py`: directional interacting-latent trainer;
- `train_plain_ce.py`: controlled plain single-head CE baseline trainer;

- dataset loading and batching;
- deterministic train/validation/test splitting;
- pointwise, ranking, and regularization losses;
- optimizer, scheduler, checkpoint, and W&B orchestration.

Run trainer implementations directly from this directory. Model architecture
code stays in `ce5/modeling/` so architecture experiments can share data,
loss, and evaluation behavior.
