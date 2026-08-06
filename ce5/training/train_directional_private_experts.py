"""Train the CE5 directional matcher with private latent scorers."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ce5.training.train_independent_latent_heads import (
    DIRECTIONAL_PRIVATE_ARCHITECTURE_TYPE,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main(architecture_type=DIRECTIONAL_PRIVATE_ARCHITECTURE_TYPE))
