"""Fixed-width ablations of the versioned Pressure state."""
from __future__ import annotations

import numpy as np

REMOVED = {
    'full': (),
    'local': (2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20),
    'without_a': tuple(range(2, 9)),
    'without_b': tuple(range(9, 15)),
    'without_c': tuple(range(15, 21)),
}


def mask_state(state: np.ndarray, variant: str) -> np.ndarray:
    """Return a same-width state with the prescribed feature group removed.

    :param state: One state or batch with 21 features in its last dimension.
    :param variant: Full, local, without_a, without_b or without_c.
    :return: Independent array with the requested feature indices zeroed.
    """
    if variant not in REMOVED:
        raise ValueError(f'Unknown ablation: {variant}')
    values = np.asarray(state)
    if values.ndim < 1 or values.shape[-1] != 21:
        raise ValueError('Expected 21 features')
    result = values.copy()
    result[..., list(REMOVED[variant])] = 0
    return result
