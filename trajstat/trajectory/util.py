"""Port of the Java ``TrajUtil`` helper methods to Python."""
from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from ..vector import Field, VectorLayer
from .config import TrajConfig


class DistanceType(str, Enum):
    """Distance metrics supported by clustering routines."""

    EUCLIDEAN = "euclidean"
    ANGLE = "angle"


@dataclass
class TrajectoryPoint:
    """Simple point representation used in distance calculations."""

    x: float
    y: float
    z: float
    m: float


# ---------------------------------------------------------------------------
# Trajectory execution and conversion utilities

def traj_cal(traj_config: TrajConfig) -> None:
    """Execute the trajectory model for every configured start time."""

    work_dir = Path(traj_config.get_traj_execute_file_name()).resolve().parent
    control_file = work_dir / "CONTROL"
    executable = Path(traj_config.get_traj_execute_file_name())
    if not executable.exists():
        raise FileNotFoundError(f"Trajectory executable not found: {executable}")

    for day_index in range(traj_config.get_day_num()):
        for hour_index in range(traj_config.get_start_hours_num()):
            traj_config.update_start_time(day_index, hour_index)
            traj_config.save_control_file(control_file)
            subprocess.run([str(executable)], cwd=work_dir, check=True)


def _iter_tgs_rows(traj_file: Path) -> Iterator[List[str]]:
    """Yield converted CSV rows for a single HYSPLIT trajectory output file."""

    with traj_file.open("r", encoding="utf-8") as reader:
        header = reader.readline()
        if not header:
            return
        meteo_file_count = int(header.strip().split()[0])
        for _ in range(meteo_file_count):
            reader.readline()

        traj_count_line = reader.readline()
        if not traj_count_line:
            return
        traj_count = int(traj_count_line.strip().split()[0])

        start_dates: List[str] = []
        for _ in range(traj_count):
            parts = reader.readline().strip().split()
            start_dates.append(
                ",".join(part.zfill(2) if len(part) < 2 else part for part in parts[:4])
            )

        reader.readline()  # skip record 5

        trajectories: List[List[List[str]]] = [[] for _ in range(traj_count)]
        for raw_line in reader:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 13:
                raise ValueError(f"Invalid trajectory line: {line}")
            idx = int(parts[0]) - 1
            w_year, w_month, w_day, w_hour = parts[2:6]
            age_hour, lat, lon, height, press = parts[8:13]
            trajectories[idx].append(
                [
                    *start_dates[idx].split(","),
                    w_year,
                    w_month,
                    w_day,
                    w_hour,
                    age_hour,
                    lat,
                    lon,
                    height,
                    press,
                ]
            )

        for trajectory in trajectories:
            for row in trajectory:
                yield row


def _write_tgs_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "start_year",
            "start_month",
            "start_day",
            "start_hour",
            "year",
            "month",
            "day",
            "hour",
            "age_hour",
            "latitude",
            "longitude",
            "height",
            "press",
        ]
    )


def traj_to_tgs(traj_file: str | Path, tgs_file: str | Path) -> None:
    """Convert a single trajectory endpoint file into the TGS CSV layout."""

    traj_path = Path(traj_file)
    if not traj_path.exists():
        return

    tgs_path = Path(tgs_file)
    tgs_path.parent.mkdir(parents=True, exist_ok=True)

    with tgs_path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.writer(writer)
        _write_tgs_header(csv_writer)
        for row in _iter_tgs_rows(traj_path):
            csv_writer.writerow(row)


def traj_to_tgs_batch(traj_files: Iterable[str | Path], tgs_file: str | Path) -> None:
    tgs_path = Path(tgs_file)
    tgs_path.parent.mkdir(parents=True, exist_ok=True)
    with tgs_path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.writer(writer)
        _write_tgs_header(csv_writer)
        for traj_file in traj_files:
            traj_path = Path(traj_file)
            if not traj_path.exists():
                continue
            for row in _iter_tgs_rows(traj_path):
                csv_writer.writerow(row)


def traj_to_tgs_from_config(traj_config: TrajConfig) -> None:
    date_format = "%Y%m%d"
    for day_index in range(traj_config.get_day_num()):
        traj_config.update_start_time(day_index, 0)
        tgs_file = (
            traj_config.out_path
            / f"{traj_config.start_time.strftime(date_format)}.tgs"
        )
        traj_files: List[Path] = []
        for hour_index in range(traj_config.get_start_hours_num()):
            traj_config.update_start_time(day_index, hour_index)
            traj_files.append(traj_config.out_path / traj_config.traj_file_name)
        traj_to_tgs_batch(traj_files, tgs_file)


def join_tgs_files_from_config(traj_config: TrajConfig) -> Path:
    month_format = "%Y%m"
    output_file = traj_config.out_path / f"{traj_config.start_time.strftime(month_format)}.tgs"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.writer(writer)
        _write_tgs_header(csv_writer)

        day_format = "%Y%m%d"
        for day_index in range(traj_config.get_day_num()):
            traj_config.update_start_time(day_index, 0)
            tgs_file = traj_config.out_path / f"{traj_config.start_time.strftime(day_format)}.tgs"
            if not tgs_file.exists():
                continue
            with tgs_file.open("r", encoding="utf-8") as reader:
                reader.readline()  # discard header
                csv_reader = csv.reader(reader)
                csv_writer.writerows(csv_reader)

    return output_file


def join_tgs_files(tgs_files: Iterable[str | Path], joined_file: str | Path) -> None:
    joined_path = Path(joined_file)
    joined_path.parent.mkdir(parents=True, exist_ok=True)
    with joined_path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.writer(writer)
        _write_tgs_header(csv_writer)
        for tgs_file in tgs_files:
            path = Path(tgs_file)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as reader:
                reader.readline()
                csv_reader = csv.reader(reader)
                csv_writer.writerows(csv_reader)


def convert_to_shape_file(tgs_file: str | Path, shp_file: str | Path) -> VectorLayer | None:
    tgs_path = Path(tgs_file)
    if not tgs_path.exists():
        return None

    layer = VectorLayer(layer_name=tgs_path.stem, shape_type="POLYLINE_Z")
    for name in ["ID", "Date", "Year", "Month", "Day", "Hour", "Height"]:
        layer.edit_add_field(Field(name=name, data_type="any"))

    with tgs_path.open("r", encoding="utf-8") as reader:
        csv_reader = csv.DictReader(reader)
        current_points: List[TrajectoryPoint] = []
        current_date: datetime | None = None
        current_height: float = 0.0
        shape_index = 0
        for row in csv_reader:
            age_hour = row["age_hour"]
            start_year = int(row["start_year"])
            start_year += 2000 if start_year < 40 else 1900
            if age_hour == "0.0" and current_points:
                layer.edit_insert_shape(current_points, layer.get_shape_num())
                layer.edit_cell_value("ID", shape_index, shape_index + 1)
                layer.edit_cell_value("Date", shape_index, current_date)
                layer.edit_cell_value("Year", shape_index, current_date.year)
                layer.edit_cell_value("Month", shape_index, current_date.month)
                layer.edit_cell_value("Day", shape_index, current_date.day)
                layer.edit_cell_value("Hour", shape_index, current_date.hour)
                layer.edit_cell_value("Height", shape_index, current_height)
                shape_index += 1
                current_points = []

            month = int(row["start_month"])
            day = int(row["start_day"])
            hour = int(row["start_hour"])
            current_date = datetime(start_year, month, day, hour)
            current_height = float(row["height"])
            point = TrajectoryPoint(
                x=float(row["longitude"]),
                y=float(row["latitude"]),
                z=float(row["height"]),
                m=float(row["press"]),
            )
            if current_points and abs(point.x - current_points[-1].x) > 100:
                if point.x > current_points[-1].x:
                    point = TrajectoryPoint(point.x - 360, point.y, point.z, point.m)
                else:
                    point = TrajectoryPoint(point.x + 360, point.y, point.z, point.m)
            current_points.append(point)

        if current_points and current_date is not None:
            layer.edit_insert_shape(current_points, layer.get_shape_num())
            layer.edit_cell_value("ID", shape_index, shape_index + 1)
            layer.edit_cell_value("Date", shape_index, current_date)
            layer.edit_cell_value("Year", shape_index, current_date.year)
            layer.edit_cell_value("Month", shape_index, current_date.month)
            layer.edit_cell_value("Day", shape_index, current_date.day)
            layer.edit_cell_value("Hour", shape_index, current_date.hour)
            layer.edit_cell_value("Height", shape_index, current_height)

    if layer.get_shape_num() == 0:
        return None

    layer.set_file_name(shp_file)
    layer.save_file()
    return layer


def remove_traj_files(traj_config: TrajConfig) -> None:
    for day_index in range(traj_config.get_day_num()):
        for hour_index in range(traj_config.get_start_hours_num()):
            traj_config.update_start_time(day_index, hour_index)
            path = traj_config.out_path / traj_config.traj_file_name
            if path.exists():
                path.unlink()


# ---------------------------------------------------------------------------
# Statistical utilities

def cal_mean_trajs(
    clusters: Sequence[int],
    cluster_level: int,
    point_num: int,
    layers: Sequence[VectorLayer],
) -> List[List[float]]:
    total_columns = point_num * 3
    traj_data = [[0.0 for _ in range(total_columns)] for _ in range(cluster_level)]
    traj_counts = [0 for _ in range(cluster_level)]

    cluster_index = 0
    for layer in layers:
        for shape in layer.get_shapes():
            if len(shape) != point_num:
                continue
            cluster = clusters[cluster_index] - 1
            cluster_index += 1
            m = 0
            for point in shape:
                traj_data[cluster][m] += point.y
                m += 1
                traj_data[cluster][m] += point.x
                m += 1
                traj_data[cluster][m] += point.z
                m += 1
            traj_counts[cluster] += 1

    for i, count in enumerate(traj_counts):
        if count == 0:
            continue
        traj_data[i] = [value / count for value in traj_data[i]]

    return traj_data


def cal_tsv(
    clusters: Sequence[int],
    cluster_level: int,
    point_num: int,
    layers: Sequence[VectorLayer],
    distance_type: DistanceType = DistanceType.EUCLIDEAN,
) -> float:
    mean_trajs = cal_mean_trajs(clusters, cluster_level, point_num, layers)
    total_variance = 0.0
    cluster_index = 0
    for layer in layers:
        for shape in layer.get_shapes():
            if len(shape) != point_num:
                continue
            cluster = clusters[cluster_index] - 1
            cluster_index += 1
            traj_a = list(shape)
            traj_b: List[TrajectoryPoint] = []
            row = mean_trajs[cluster]
            m = 0
            for _ in range(point_num):
                y = row[m]
                m += 1
                x = row[m]
                m += 1
                z = row[m]
                m += 1
                traj_b.append(TrajectoryPoint(x, y, z, 0.0))
            if distance_type == DistanceType.EUCLIDEAN:
                total_variance += cal_distance_euclidean(traj_a, traj_b)
            else:
                total_variance += cal_distance_angle(traj_a, traj_b)

    return total_variance


def cal_distance_euclidean(
    traj_a: Sequence[TrajectoryPoint], traj_b: Sequence[TrajectoryPoint]
) -> float:
    dist = 0.0
    for point_a, point_b in zip(traj_a, traj_b):
        dist += (point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2
    return math.sqrt(dist)


def cal_distance_angle(
    traj_a: Sequence[TrajectoryPoint], traj_b: Sequence[TrajectoryPoint]
) -> float:
    dist = 0.0
    if not traj_a or not traj_b:
        return dist
    x0 = traj_a[0].x
    y0 = traj_b[0].y
    for point_a, point_b in zip(traj_a[1:], traj_b[1:]):
        a = (point_a.x - x0) ** 2 + (point_a.y - y0) ** 2
        b = (point_b.x - x0) ** 2 + (point_b.y - y0) ** 2
        c = (point_b.x - point_a.x) ** 2 + (point_b.y - point_a.y) ** 2
        if a == 0 or b == 0:
            angle = 0.0
        else:
            angle = 0.5 * (a + b - c) / math.sqrt(a * b)
        angle = max(min(angle, 1.0), -1.0)
        dist += math.acos(angle)
    return dist / len(traj_a)


# ---------------------------------------------------------------------------
# Miscellaneous helpers

def get_time_zone(time_zone_str: str) -> int:
    time_zone_str = time_zone_str.strip()
    offset = time_zone_str[3:]
    if offset.startswith("+"):
        offset = offset[1:]
    return int(offset)


def add_data_to_traj(
    data_file: str | Path,
    start_date_index: int,
    format_str: str,
    time_zone: int,
    data_field_index: int,
    undef: float,
    layers: Sequence[VectorLayer],
    field_name: str,
    data_type: str,
    field_length: int,
    field_decimal: int,
) -> None:
    entries: List[tuple[str, str]] = []
    with Path(data_file).open("r", encoding="utf-8") as reader:
        csv_reader = csv.reader(reader)
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) <= max(start_date_index, data_field_index):
                continue
            entries.append((row[start_date_index], row[data_field_index]))

    for layer in layers:
        field_idx = layer.get_field_idx_by_name(field_name)
        if field_idx == -1:
            layer.edit_add_field(field_name, data_type, field_length, field_decimal)
            field_idx = layer.get_field_number() - 1
        shape_count = layer.get_shape_num()
        date_field_index = layer.get_field_idx_by_name("Date")
        for i in range(shape_count):
            if date_field_index > -1:
                dt = layer.get_cell_value("Date", i)
                if isinstance(dt, datetime):
                    dt = dt.replace(hour=int(layer.get_cell_value("Hour", i)))
                else:
                    hour = int(layer.get_cell_value("Hour", i))
                    dt = datetime.strptime(str(dt), "%Y-%m-%d %H:%M:%S")
                    dt = dt.replace(hour=hour)
                dt = dt + timedelta(hours=time_zone)
            else:
                year = int(layer.get_cell_value("Year", i))
                month = int(layer.get_cell_value("Month", i))
                day = int(layer.get_cell_value("Day", i))
                hour = int(layer.get_cell_value("Hour", i))
                dt = datetime(year, month, day, hour) + timedelta(hours=time_zone)
            date_str = dt.strftime(format_str)
            layer.edit_cell_value(field_idx, i, _default_value(data_type, undef))
            for entry_date, entry_value in entries:
                if date_str.startswith(entry_date):
                    layer.edit_cell_value(
                        field_idx, i, _cast_value(data_type, entry_value, undef)
                    )
                    break
        layer.get_attribute_table().save()


def _default_value(data_type: str, undef: float) -> object:
    if data_type.lower() == "int":
        return int(undef)
    if data_type.lower() == "string":
        return "Null"
    return undef


def _cast_value(data_type: str, value: str, undef: float) -> object:
    if data_type.lower() == "int":
        return int(value) if value else int(undef)
    if data_type.lower() == "double":
        return float(value) if value else float(undef)
    return value
