from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, Iterable, List


def expand_search_space(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, combo)})
    return combos


def sample_trials(
    space: Dict[str, List[Any]],
    max_trials: int,
    seed: int,
) -> List[Dict[str, Any]]:
    combos = expand_search_space(space)
    if len(combos) <= max_trials:
        return combos
    rng = random.Random(seed)
    return rng.sample(combos, max_trials)
