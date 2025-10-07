from __future__ import annotations

import csv
from pathlib import Path

import pytest

from trajstat.vector import VectorLayer

from trajstat.trajectory.util import (
    DistanceType,
    TrajectoryPoint,
    add_data_to_traj,
    cal_mean_trajs,
    cal_tsv,
    calculate_cwt_field,
    calculate_pscf_field,
    convert_to_shape_file,
    get_time_zone,
    populate_endpoint_counts,
    traj_to_tgs,
    weight_by_counts,
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


def _build_test_layers() -> tuple[VectorLayer, VectorLayer]:
    polygon_layer = VectorLayer("grid", "POLYGON")
    polygon_layer.edit_insert_shape(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], 0
    )
    polygon_layer.edit_insert_shape(
        [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)], 1
    )

    traj_layer = VectorLayer("traj", "POLYLINE_Z")
    traj_layer.edit_add_field("Value", "double")
    traj_layer.edit_insert_shape(
        [
            TrajectoryPoint(0.25, 0.25, 50.0, 0.0),
            TrajectoryPoint(0.75, 0.75, 50.0, 0.0),
        ],
        0,
    )
    traj_layer.edit_cell_value("Value", 0, 5.0)
    traj_layer.edit_insert_shape(
        [
            TrajectoryPoint(0.75, 0.75, 10.0, 0.0),
            TrajectoryPoint(1.25, 0.75, 10.0, 0.0),
        ],
        1,
    )
    traj_layer.edit_cell_value("Value", 1, 50.0)
    return polygon_layer, traj_layer


def test_pscf_workflow() -> None:
    polygon_layer, traj_layer = _build_test_layers()
    nij, traj_counts = populate_endpoint_counts(
        [traj_layer],
        polygon_layer,
        value_field="Value",
        missing_value=-9999.0,
        count_field="Nij",
        trajectory_count_field="N_Traj",
    )
    assert nij == [3, 1]
    assert traj_counts == [2, 1]

    mij, _ = populate_endpoint_counts(
        [traj_layer],
        polygon_layer,
        value_field="Value",
        missing_value=-9999.0,
        count_field="Mij",
        criterion=20.0,
    )
    assert mij == [1, 1]

    pscf = calculate_pscf_field(
        polygon_layer, nij_field="Nij", mij_field="Mij", output_field="PSCF"
    )
    assert pytest.approx(pscf[0]) == pytest.approx(1 / 3)
    assert pytest.approx(pscf[1]) == pytest.approx(1.0)

    weighted = weight_by_counts(
        polygon_layer,
        base_field="PSCF",
        count_field="Nij",
        target_field="WPSCF",
        thresholds=[2, 1],
        ratios=[0.5, 0.2],
    )
    assert pytest.approx(weighted[0]) == pytest.approx(1 / 3)
    assert pytest.approx(weighted[1]) == pytest.approx(0.2)

    weighted_traj = weight_by_counts(
        polygon_layer,
        base_field="WPSCF",
        count_field="N_Traj",
        target_field="WPSCF2",
        thresholds=[1],
        ratios=[0.1],
    )
    assert pytest.approx(weighted_traj[0]) == pytest.approx(1 / 3)
    assert pytest.approx(weighted_traj[1]) == pytest.approx(0.02)


def test_cwt_calculation() -> None:
    polygon_layer, traj_layer = _build_test_layers()
    cwts, counts = calculate_cwt_field(
        [traj_layer],
        polygon_layer,
        value_field="Value",
        missing_value=-9999.0,
        output_field="CWT",
        count_field="Nij_CWT",
    )
    assert counts == [3, 1]
    assert pytest.approx(cwts[0]) == pytest.approx(20.0)
    assert pytest.approx(cwts[1]) == pytest.approx(50.0)

    weighted = weight_by_counts(
        polygon_layer,
        base_field="CWT",
        count_field="Nij_CWT",
        target_field="WCWT",
        thresholds=[2, 1],
        ratios=[0.5, 0.2],
    )
    assert pytest.approx(weighted[0]) == pytest.approx(20.0)
    assert pytest.approx(weighted[1]) == pytest.approx(10.0)
