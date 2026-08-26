"""Global seed control.

The chunking pipeline itself is deterministic by construction - it performs no
sampling - so seeding mainly guards against accidental nondeterminism; the
evaluation's bootstrap statistics are where the seed genuinely matters. Every
RNG is seeded and the seed is recorded in the run manifest so the whole
pipeline is reproducible end to end.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedState:
    """Snapshot of what was seeded, for the run manifest."""

    seed: int
    pythonhashseed: str | None
    numpy_seeded: bool


def set_global_seed(seed: int = 20260628) -> SeedState:
    """Seed all RNGs we can reach and return a snapshot.

    Note on ``PYTHONHASHSEED``: Python reads it only at interpreter startup, so
    setting it here does not change hash randomization for the *current*
    process. For fully reproducible hash ordering, export it before launching
    (``PYTHONHASHSEED=0 python ...``). The pipeline deliberately avoids relying
    on set/dict iteration order, so this does not affect chunk output; we set
    and record it for completeness and for any subprocess we spawn.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except Exception:  # numpy optional at seed time
        pass

    # torch hook: torch.manual_seed / torch.cuda.manual_seed_all could be added
    # here without changing this module's interface. Not needed today - the
    # embedders only run inference, which torch keeps deterministic per device.

    return SeedState(
        seed=seed,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        numpy_seeded=numpy_seeded,
    )
