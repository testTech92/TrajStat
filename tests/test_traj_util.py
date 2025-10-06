from __future__ import annotations

import csv
from pathlib import Path

import pytest

from trajstat.trajectory.util import (
    DistanceType,
    add_data_to_traj,
    cal_mean_trajs,
    cal_tsv,
    convert_to_shape_file,
    get_time_zone,
    traj_to_tgs,
)


def _write_sample_traj(path: Path) -> None:
    lines = [
        "1 1 1\n",
        "dummy\n",
        "1\n",
        "19 1 1 0\n",
        "1\n",
        " 1 1 19 1 1 0 0 0 0.0 30.0 100.0 1500.0 900.0\n",
        " 1 1 19 1 1 1 0 0 -1.0 31.0 101.0 1400.0 905.0\n",
    ]
    path.write_text("".join(lines))


def test_traj_to_tgs_and_convert(tmp_path: Path) -> None:
    traj_file = tmp_path / "sample.traj"
    _write_sample_traj(traj_file)
    tgs_file = tmp_path / "sample.tgs"
    traj_to_tgs(traj_file, tgs_file)

    with tgs_file.open() as reader:
        rows = list(csv.DictReader(reader))
    assert len(rows) == 2
    assert rows[0]["latitude"] == "30.0"

    geojson = tmp_path / "sample.geojson"
    layer = convert_to_shape_file(tgs_file, geojson)
    assert layer is not None
    assert layer.get_shape_num() == 1
    shape = layer.get_shapes()[0]
    assert pytest.approx(shape[0].y) == 30.0
    assert pytest.approx(shape[1].x) == 101.0

    means = cal_mean_trajs([1], 1, 2, [layer])
    assert pytest.approx(means[0][0]) == 30.0
    tsv = cal_tsv([1], 1, 2, [layer], DistanceType.EUCLIDEAN)
    assert pytest.approx(tsv) == 0.0


def test_add_data_to_traj(tmp_path: Path) -> None:
    traj_file = tmp_path / "sample.traj"
    _write_sample_traj(traj_file)
    tgs_file = tmp_path / "sample.tgs"
    traj_to_tgs(traj_file, tgs_file)
    layer = convert_to_shape_file(tgs_file, tmp_path / "sample.geojson")
    assert layer is not None

    data_file = tmp_path / "data.csv"
    data_file.write_text("date,value\n2019010100,42.5\n")

    add_data_to_traj(
        data_file,
        start_date_index=0,
        format_str="%Y%m%d%H",
        time_zone=0,
        data_field_index=1,
        undef=-9999.0,
        layers=[layer],
        field_name="PM25",
        data_type="double",
        field_length=10,
        field_decimal=2,
    )

    value = layer.get_cell_value("PM25", 0)
    assert pytest.approx(value) == 42.5


def test_get_time_zone() -> None:
    assert get_time_zone("GMT+4") == 4
    assert get_time_zone("GMT-3") == -3
