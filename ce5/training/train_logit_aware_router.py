"""Train the CE5 independent-v2 logit-aware router experiment."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ce5.training.train_independent_latent_heads import (
    LOGIT_AWARE_ROUTER_ARCHITECTURE_TYPE,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main(architecture_type=LOGIT_AWARE_ROUTER_ARCHITECTURE_TYPE))
