"""EV data container and npz reader (port of ``JULIA/EV.jl``)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np


@dataclass
class EVData:
    """Parameters of K electric vehicles (EVs); every field is a 1-D array of length K.

    ``ta`` is the first time step an EV is connected (included) and ``td`` is the step at which it leaves
    (excluded), both on the absolute time grid of the scenario.
    """

    ta: np.ndarray  # arrival step (included), integer
    td: np.ndarray  # departure step (excluded), integer
    SoCa: np.ndarray  # SoC at arrival
    SoCd: np.ndarray  # desired SoC at departure
    C: np.ndarray  # battery capacity [kWh]
    e: np.ndarray  # charging efficiency
    SoCmin: np.ndarray  # lower SoC bound
    SoCmax: np.ndarray  # upper SoC bound
    Pmax: np.ndarray  # maximum charging power [kW]

    def __post_init__(self) -> None:
        """Cast all fields to 1-D arrays: ``ta`` and ``td`` to int64, the others to float64.

        The npz files store ``ta`` and ``td`` as float64, whereas the Julia struct required ``Vector{Int}``.

        :raises ValueError: If ``ta`` or ``td`` contains non-integral values.
        """
        for f in fields(self):
            x = np.asarray(getattr(self, f.name))
            if f.name in ("ta", "td"):
                if not np.array_equal(x, np.rint(x)):
                    raise ValueError(f"{f.name} must be integral")
                x = np.rint(x).astype(np.int64)
            else:
                x = x.astype(np.float64)
            setattr(self, f.name, np.atleast_1d(x))

    def __len__(self) -> int:
        """Return the number of EVs.

        :return: The number of EVs K.
        """
        return len(self.ta)

    @classmethod
    def from_npz(cls, npz_file: str | Path) -> EVData:
        """Read the EV parameters from a scenario npz file.

        :param npz_file: Path of the scenario file; it must contain one key per field.
        :return: The EV parameters.
        """
        with np.load(npz_file) as data:
            return cls(**{f.name: data[f.name] for f in fields(cls)})

    def take(self, idx: slice | list[int] | np.ndarray) -> EVData:
        """Select a subset of the EVs.

        :param idx: Slice or indices of the EVs to keep.
        :return: A new :class:`EVData` that contains only the selected EVs.
        """
        return EVData(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})
