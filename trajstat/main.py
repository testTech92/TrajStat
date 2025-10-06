"""Python port of the TrajStat plug-in entry point."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .trajectory.config import TrajConfig
from .trajectory.util import (
    DistanceType,
    add_data_to_traj,
    cal_mean_trajs,
    cal_tsv,
    convert_to_shape_file,
    get_time_zone,
    join_tgs_files,
    join_tgs_files_from_config,
    traj_cal,
    traj_to_tgs,
    traj_to_tgs_batch,
    traj_to_tgs_from_config,
)
from .vector import VectorLayer


@dataclass
class TrajStatPlugin:
    """High level façade that mirrors the historical Java plug-in API."""

    config: TrajConfig = field(default_factory=TrajConfig)

    def load_control(self, control_file: str | Path) -> None:
        self.config.load_control_file(control_file)

    def calculate_trajectories(self) -> None:
        traj_cal(self.config)

    def convert_to_tgs(self) -> None:
        traj_to_tgs_from_config(self.config)

    def join_tgs_files(self) -> Path:
        return join_tgs_files_from_config(self.config)

    def convert_to_shape(self, tgs_file: str | Path, output_file: str | Path) -> VectorLayer | None:
        return convert_to_shape_file(tgs_file, output_file)

    def append_measurements(
        self,
        data_file: str | Path,
        start_date_index: int,
        format_str: str,
        time_zone: str,
        data_field_index: int,
        undef: float,
        layers: Sequence[VectorLayer],
        field_name: str,
        data_type: str,
        field_length: int,
        field_decimal: int,
    ) -> None:
        tz = get_time_zone(time_zone)
        add_data_to_traj(
            data_file,
            start_date_index,
            format_str,
            tz,
            data_field_index,
            undef,
            layers,
            field_name,
            data_type,
            field_length,
            field_decimal,
        )

    def calculate_cluster_statistics(
        self,
        clusters: Sequence[int],
        cluster_level: int,
        point_num: int,
        layers: Sequence[VectorLayer],
        distance: DistanceType = DistanceType.EUCLIDEAN,
    ) -> tuple[List[List[float]], float]:
        means = cal_mean_trajs(clusters, cluster_level, point_num, layers)
        tsv = cal_tsv(clusters, cluster_level, point_num, layers, distance)
        return means, tsv

    # Convenience wrappers for direct file based conversions -----------------
    @staticmethod
    def convert_single_traj(traj_file: str | Path, tgs_file: str | Path) -> None:
        traj_to_tgs(traj_file, tgs_file)

    @staticmethod
    def convert_multiple_traj(traj_files: Iterable[str | Path], tgs_file: str | Path) -> None:
        traj_to_tgs_batch(traj_files, tgs_file)

    @staticmethod
    def merge_tgs_files(tgs_files: Iterable[str | Path], joined_file: str | Path) -> None:
        join_tgs_files(tgs_files, joined_file)
