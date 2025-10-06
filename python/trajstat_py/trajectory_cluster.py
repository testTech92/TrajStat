"""Clustering utilities for back trajectory analyses.

This module rewrites the Java based clustering logic used by TrajStat into
pure Python code.  The :class:`TrajectoryClusterer` class provides a minimal
k-means implementation that works with evenly sampled trajectories and can be
used to reproduce the cluster statistics produced by the desktop
application.  The implementation also exposes helpers to compute the cluster
mean trajectories and total spatial variance (TSV) using both Euclidean and
angular distance metrics, mirroring the original Java algorithms.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = ["TrajectoryClusterer"]


class TrajectoryClusterer:
    """Cluster back trajectories using a simple k-means implementation.

    Parameters
    ----------
    trajectories:
        Iterable of trajectories.  Each trajectory must be convertible to a
        ``numpy.ndarray`` of shape ``(n_points, n_dimensions)``.  The clusterer
        assumes that every trajectory contains the same number of sampled
        points, which matches the behaviour of the original TrajStat code.

    Attributes
    ----------
    centroids_:
        Array with shape ``(n_clusters, n_points, n_dimensions)`` containing the
        cluster centroids (mean trajectories).
    labels_:
        Cluster index assigned to every trajectory.
    """

    def __init__(self, trajectories: Iterable[Sequence[Sequence[float]]]):
        trajs = [np.asarray(traj, dtype=float) for traj in trajectories]
        if not trajs:
            raise ValueError("At least one trajectory is required.")

        lengths = {traj.shape for traj in trajs}
        if len(lengths) != 1:
            raise ValueError("All trajectories must have the same shape.")

        self._trajectories = np.stack(trajs, axis=0)
        self._n_samples, self._n_points, self._n_dims = self._trajectories.shape
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        n_clusters: int,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> "TrajectoryClusterer":
        """Run k-means on the trajectory set.

        The algorithm clusters flattened trajectory coordinates.  Although
        simple, this mirrors the original workflow where the spatial variance
        is measured using Euclidean or angular metrics after clustering.

        Parameters
        ----------
        n_clusters:
            Number of clusters to form.
        max_iter:
            Maximum number of k-means iterations.
        tol:
            Convergence tolerance based on centroid displacement.
        random_state:
            Optional seed for deterministic initial centroid selection.
        """

        if n_clusters <= 1:
            raise ValueError("n_clusters must be greater than 1.")
        if n_clusters > self._n_samples:
            raise ValueError("n_clusters must be <= number of trajectories.")

        rng = np.random.default_rng(random_state)
        flat_trajs = self._trajectories.reshape(self._n_samples, -1)

        # Initialise centroids using a simple random choice without replacement
        # to stay dependency-free.
        indices = rng.choice(self._n_samples, size=n_clusters, replace=False)
        centroids = flat_trajs[indices]

        labels = np.zeros(self._n_samples, dtype=int)
        for _ in range(max_iter):
            distances = np.linalg.norm(flat_trajs[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = distances.argmin(axis=1)

            new_centroids = np.empty_like(centroids)
            for k in range(n_clusters):
                members = flat_trajs[new_labels == k]
                if members.size == 0:
                    # Reinitialise empty clusters using a random trajectory.
                    new_centroids[k] = flat_trajs[rng.integers(self._n_samples)]
                else:
                    new_centroids[k] = members.mean(axis=0)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels = new_labels
            if shift < tol:
                break

        self.centroids_ = centroids.reshape(n_clusters, self._n_points, self._n_dims)
        self.labels_ = labels
        return self

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------
    def mean_trajectories(self) -> np.ndarray:
        """Return the mean trajectory for every cluster."""

        self._ensure_fitted()
        assert self.centroids_ is not None
        return self.centroids_.copy()

    def cluster_sizes(self) -> np.ndarray:
        """Return the number of trajectories contained in every cluster."""

        self._ensure_fitted()
        assert self.labels_ is not None
        counts = np.bincount(self.labels_, minlength=self.centroids_.shape[0])
        return counts.astype(int)

    def cluster_ratios(self) -> np.ndarray:
        """Return the fraction of trajectories assigned to every cluster."""

        counts = self.cluster_sizes()
        return counts / counts.sum()

    # ------------------------------------------------------------------
    # Total spatial variance
    # ------------------------------------------------------------------
    def total_spatial_variance(self, metric: str = "euclidean") -> float:
        """Compute the TSV of the clustering using the requested metric."""

        self._ensure_fitted()
        assert self.labels_ is not None and self.centroids_ is not None

        metric = metric.lower()
        if metric not in {"euclidean", "angle"}:
            raise ValueError("metric must be 'euclidean' or 'angle'.")

        variance = 0.0
        for idx, traj in enumerate(self._trajectories):
            centroid = self.centroids_[self.labels_[idx]]
            if metric == "euclidean":
                variance += self._euclidean_distance(traj, centroid)
            else:
                variance += self._angle_distance(traj, centroid)
        return variance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_fitted(self) -> None:
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("Call 'fit' before requesting cluster statistics.")

    @staticmethod
    def _euclidean_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
        diff = (traj_a - traj_b)[..., :2]
        return float(np.sqrt(np.sum(diff ** 2)))

    @staticmethod
    def _angle_distance(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
        # The Java implementation ignores the vertical coordinate and compares
        # the horizontal components of the trajectories.
        xy_a = traj_a[:, :2]
        xy_b = traj_b[:, :2]
        x0, y0 = xy_a[0]
        angles = []
        for point_a, point_b in zip(xy_a[1:], xy_b[1:]):
            a = np.sum((point_a - (x0, y0)) ** 2)
            b = np.sum((point_b - (x0, y0)) ** 2)
            c = np.sum((point_b - point_a) ** 2)
            if a == 0 or b == 0:
                cosine = 0.0
            else:
                cosine = 0.5 * (a + b - c) / np.sqrt(a * b)
                cosine = float(np.clip(cosine, -1.0, 1.0))
            angles.append(np.arccos(cosine))
        if not angles:
            return 0.0
        return float(np.sum(angles) / xy_a.shape[0])
