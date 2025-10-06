"""Concentration Weighted Trajectory (CWT) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .pscf import GridDefinition

__all__ = ["CWTCalculator", "CWTResult"]


@dataclass
class CWTResult:
    nij: np.ndarray
    sum_values: np.ndarray
    cwt: np.ndarray


class CWTCalculator:
    """Calculate CWT values on a regular analysis grid."""

    def __init__(
        self,
        grid: GridDefinition,
        trajectories: Iterable[Sequence[Sequence[float]]],
        measurements: Sequence[float],
        *,
        missing_value: float = np.nan,
    ) -> None:
        self.grid = grid
        self.trajectories = [np.asarray(traj, dtype=float) for traj in trajectories]
        self.measurements = np.asarray(measurements, dtype=float)
        if len(self.trajectories) != len(self.measurements):
            raise ValueError("Each trajectory must have an associated measurement value.")
        self.missing_value = missing_value

    def compute(self, *, altitude_threshold: Optional[float] = None) -> CWTResult:
        nij = np.zeros(self.grid.shape, dtype=int)
        sums = np.zeros_like(nij, dtype=float)

        for traj, value in zip(self.trajectories, self.measurements):
            if self._is_missing(value):
                continue
            for point in traj:
                if altitude_threshold is not None and point.shape[0] >= 3 and point[2] > altitude_threshold:
                    continue
                ix, iy = self._point_to_index(point)
                if ix is None:
                    continue
                nij[iy, ix] += 1
                sums[iy, ix] += value

        with np.errstate(divide="ignore", invalid="ignore"):
            cwt = np.divide(sums, nij, out=np.zeros_like(sums), where=nij > 0)
        return CWTResult(nij=nij, sum_values=sums, cwt=cwt)

    def apply_weighting(
        self,
        data: np.ndarray,
        *,
        thresholds: Sequence[int],
        weights: Sequence[float],
        counts: np.ndarray,
    ) -> np.ndarray:
        from .pscf import PSCFCalculator

        return PSCFCalculator.apply_weighting(
            data,
            thresholds=thresholds,
            weights=weights,
            counts=counts,
        )

    def _point_to_index(self, point: Sequence[float]) -> Tuple[Optional[int], Optional[int]]:
        x, y = point[0], point[1]
        if not (self.grid.x_min <= x < self.grid.x_max and self.grid.y_min <= y < self.grid.y_max):
            return None, None
        ix = int((x - self.grid.x_min) / self.grid.cell_size)
        iy = int((y - self.grid.y_min) / self.grid.cell_size)
        ny, nx = self.grid.shape
        if 0 <= ix < nx and 0 <= iy < ny:
            return ix, iy
        return None, None

    def _is_missing(self, value: float) -> bool:
        if np.isnan(self.missing_value):
            return np.isnan(value)
        return float(value) == float(self.missing_value)
