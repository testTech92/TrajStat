# TrajStat (Python Edition)

TrajStat is a toolkit for working with atmospheric back trajectories.  This
repository contains a Python re-implementation of the original MeteoInfo
plug-in.  The code base provides the same high level capabilities as the Java
version – trajectory conversion, data augmentation and clustering utilities –
while embracing the Python ecosystem.

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

## Python API

The :class:`trajstat.main.TrajStatPlugin` façade offers a lightweight, fully
typed interface for embedding TrajStat functionality inside other
applications.  All trajectory specific helpers are available from
`trajstat.trajectory`.

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
