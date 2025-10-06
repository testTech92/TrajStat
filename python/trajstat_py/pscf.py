"""Potential Source Contribution Function (PSCF) implementation.

The routines below port the logic that powers the Java based PSCF dialog to a
pure Python workflow.  They operate on regular rectilinear grids and work with
arbitrary trajectory collections that include measurement data associated with
an entire trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["GridDefinition", "PSCFCalculator", "PSCFResult"]


@dataclass(frozen=True)
class GridDefinition:
    """Describe a regular analysis grid."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cell_size: float

    @property
    def x_edges(self) -> np.ndarray:
        return np.arange(self.x_min, self.x_max + self.cell_size, self.cell_size)

    @property
    def y_edges(self) -> np.ndarray:
        return np.arange(self.y_min, self.y_max + self.cell_size, self.cell_size)

    @property
    def shape(self) -> Tuple[int, int]:
        nx = int(np.ceil((self.x_max - self.x_min) / self.cell_size))
        ny = int(np.ceil((self.y_max - self.y_min) / self.cell_size))
        return ny, nx


@dataclass
class PSCFResult:
    nij: np.ndarray
    mij: np.ndarray
    pscf: np.ndarray
    traj_counts: np.ndarray


class PSCFCalculator:
    """Calculate PSCF statistics from trajectory end points."""

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

    # ------------------------------------------------------------------
    def compute(
        self,
        threshold: float,
        *,
        altitude_threshold: Optional[float] = None,
    ) -> PSCFResult:
        """Compute Nij, Mij and the PSCF field."""

        nij, traj_counts = self._count_endpoints(altitude_threshold=altitude_threshold)
        mij, _ = self._count_endpoints(
            threshold=threshold,
            altitude_threshold=altitude_threshold,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            pscf = np.divide(mij, nij, out=np.zeros_like(mij, dtype=float), where=nij > 0)
        return PSCFResult(nij=nij, mij=mij, pscf=pscf, traj_counts=traj_counts)

    # ------------------------------------------------------------------
    @staticmethod
    def apply_weighting(
        data: np.ndarray,
        *,
        thresholds: Sequence[int],
        weights: Sequence[float],
        counts: np.ndarray,
    ) -> np.ndarray:
        """Apply the empirical weighting scheme used by TrajStat."""

        if len(thresholds) != len(weights):
            raise ValueError("thresholds and weights must have the same length.")
        thresholds = np.asarray(thresholds, dtype=int)
        weights = np.asarray(weights, dtype=float)

        result = data.astype(float).copy()
        flat_counts = counts.ravel()
        flat_data = result.ravel()
        for i, count in enumerate(flat_counts):
            weight = 1.0
            for idx, threshold in enumerate(thresholds):
                if idx == len(thresholds) - 1:
                    if count <= threshold:
                        weight = weights[idx]
                        break
                else:
                    if thresholds[idx + 1] < count <= threshold:
                        weight = weights[idx]
                        break
            flat_data[i] *= weight
        return flat_data.reshape(result.shape)

    # ------------------------------------------------------------------
    def _count_endpoints(
        self,
        *,
        threshold: Optional[float] = None,
        altitude_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        counts = np.zeros(self.grid.shape, dtype=int)
        traj_counts = np.zeros_like(counts)

        for traj, value in zip(self.trajectories, self.measurements):
            if self._is_missing(value):
                continue
            if threshold is not None and value < threshold:
                continue

            visited = set()
            for point in traj:
                if altitude_threshold is not None and point.shape[0] >= 3 and point[2] > altitude_threshold:
                    continue
                ix, iy = self._point_to_index(point)
                if ix is None:
                    continue
                counts[iy, ix] += 1
                if (iy, ix) not in visited:
                    traj_counts[iy, ix] += 1
                    visited.add((iy, ix))
        return counts, traj_counts

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
