# TrajStat (Python Edition)

TrajStat is a toolkit for working with atmospheric back trajectories.  This
repository contains a Python re-implementation of the original MeteoInfo
plug-in.  The code base provides the same high level capabilities as the Java
version – trajectory conversion, data augmentation and clustering utilities –
while embracing the Python ecosystem.

> 🇨🇳 文档：请参阅 [中文 README](README.zh.md) 以获取计算公式与实现说明。

## Features

- Execute external HYSPLIT calculations using rich `TrajConfig` objects.
- Convert raw trajectory endpoint listings into the TrajStat (TGS) CSV format.
- Merge daily TGS files and transform them into GeoJSON polyline collections.
- Attach measurement data to trajectory layers and compute cluster statistics
  using Euclidean or angular distance metrics.
- Typer powered command line interface exposed as the `trajstat` executable.

## Installation

The project targets **Python 3.9+**.  Install the package and its dependencies
with pip:

```bash
pip install .
```

Development dependencies (chiefly `pytest`) can be installed via the optional
`dev` extra:

```bash
pip install .[dev]
```

## Command Line Usage

The CLI mirrors the classic workflow:

```bash
trajstat calculate CONTROL          # run HYSPLIT for every configured start time
trajstat convert sample.traj sample.tgs
trajstat convert-batch *.traj month.tgs
trajstat join day1.tgs day2.tgs month.tgs
trajstat shape month.tgs month.geojson
trajstat cluster-stats month.tgs 1 1 --point-count 24
trajstat timezone GMT+8
```

Use `trajstat --help` to inspect every command and option.

### PSCF/CWT Examples

The cluster analysis and potential source contribution features are also
available from the CLI.  The following snippet outlines a typical workflow
starting from daily trajectory CSV files and a polygon grid layer stored as
GeoJSON:

```bash
# Derive cluster means and the total spatial variance for a cluster solution
trajstat cluster-stats month.tgs 1 5 --point-count 24

# Populate Nij/Mij endpoint counts for PSCF before applying the ratios
trajstat pscf-counts month.geojson month.tgs --value-field SO2 --missing-value -9999

# Calculate PSCF values and write them back to the grid file
trajstat pscf month.geojson --nij-field Nij --mij-field Mij --output-field PSCF

# Compute concentration-weighted trajectories using the same layers
trajstat cwt month.geojson month.tgs --value-field SO2 --missing-value -9999
```

All PSCF/CWT commands accept `--help` for additional options such as
thresholded weighting and explicit trajectory count fields.

## Python API

The :class:`trajstat.main.TrajStatPlugin` façade offers a lightweight, fully
typed interface for embedding TrajStat functionality inside other
applications.  All trajectory specific helpers are available from
`trajstat.trajectory`.

```python
from pathlib import Path

from trajstat.main import TrajStatPlugin
from trajstat.vector import VectorLayer

plugin = TrajStatPlugin()

# Load previously converted trajectory layers
layers = [VectorLayer.from_geojson(Path("month.geojson"))]

# Cluster mean trajectories and TSV
means, tsv = plugin.calculate_cluster_statistics([1, 1, 2], 1, 24, layers)

# Fill PSCF endpoint counts and compute ratios
polygon = VectorLayer.from_geojson(Path("grid.geojson"))
plugin.fill_endpoint_counts(
    layers,
    polygon,
    value_field="SO2",
    missing_value=-9999,
    count_field="Nij",
    trajectory_count_field="Mij",
)
pscf = plugin.calculate_pscf(polygon, nij_field="Nij", mij_field="Mij")

# Optional CWT calculation
cwt, counts = plugin.calculate_cwt(
    layers,
    polygon,
    value_field="SO2",
    missing_value=-9999,
)
```

The `VectorLayer` helper can read GeoJSON, shapefiles, and other vector
formats supported by Fiona.

## Testing

Run the unit test suite with:

```bash
pytest
```

The tests cover the core data conversion and augmentation routines using small
sample files.

## Licensing and Credits

The original TrajStat project was authored by **Yaqiang Wang**.  This Python
port remains compatible with the GNU LGPL v2.1 (or later).
