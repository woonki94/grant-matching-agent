"""Train the CE5 reliability-aware residual router experiment.

The shared CE5 trainer handles dataset splitting and checkpoint selection.  The
selected architecture freezes independent-v2 and adds teacher-derived routing
and reliability losses without exposing labels to the model forward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ce5.training.train_independent_latent_heads import (
    RELIABILITY_AWARE_ROUTER_ARCHITECTURE_TYPE,
    main,
)


if __name__ == "__main__":
    raise SystemExit(
        main(architecture_type=RELIABILITY_AWARE_ROUTER_ARCHITECTURE_TYPE)
    )
