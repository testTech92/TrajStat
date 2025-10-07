"""Core configuration helpers shared across trajectory operations."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from .location import Location


CONTROL_DATETIME_FORMAT = "%y %m %d %H"


@dataclass
class TrajControl:
    """In-memory representation of the HYSPLIT ``CONTROL`` file."""

    start_time: datetime = field(default_factory=datetime.utcnow)
    locations: List[Location] = field(default_factory=list)
    run_hours: int = -24
    vertical: int = 0
    top_of_model: float = 10_000.0
    meteo_files: List[Path] = field(default_factory=list)
    out_path: Path = Path(".")
    traj_file_name: str = "traj"

    def __post_init__(self) -> None:
        self.out_path = Path(self.out_path)
        if self.locations and not isinstance(self.locations[0], Location):
            self.locations = [
                loc if isinstance(loc, Location) else Location(*loc)
                for loc in self.locations
            ]
        self.meteo_files = [Path(f) for f in self.meteo_files]

    # ------------------------------------------------------------------
    # Convenience helpers
    def set_location(self, location: Location | str) -> None:
        if isinstance(location, str):
            location = Location.from_string(location)
        self.locations = [location]

    def add_location(self, location: Location | str) -> None:
        if isinstance(location, str):
            location = Location.from_string(location)
        self.locations.append(location)

    # ------------------------------------------------------------------
    # Serialisation helpers
    def load_control_file(self, file_name: str | Path) -> None:
        path = Path(file_name)
        with path.open("r", encoding="utf-8") as handle:
            start_time_str = handle.readline().strip()
            self.start_time = datetime.strptime(start_time_str, CONTROL_DATETIME_FORMAT)

            loc_count = int(handle.readline().strip())
            self.locations = []
            for _ in range(loc_count):
                self.locations.append(Location.from_string(handle.readline()))

            self.run_hours = int(handle.readline().strip())
            self.vertical = int(handle.readline().strip())
            self.top_of_model = float(handle.readline().strip())

            meteo_count = int(handle.readline().strip())
            self.meteo_files = []
            for _ in range(meteo_count):
                directory = handle.readline().strip()
                filename = handle.readline().strip()
                self.meteo_files.append(Path(directory) / filename)

            self.out_path = Path(handle.readline().strip())
            self.traj_file_name = handle.readline().strip()

    def save_control_file(self, file_name: str | Path) -> None:
        path = Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(self.start_time.strftime(CONTROL_DATETIME_FORMAT) + "\n")
            handle.write(f"{len(self.locations)}\n")
            for location in self.locations:
                handle.write(location.to_control_string() + "\n")

            handle.write(f"{self.run_hours}\n")
            handle.write(f"{self.vertical}\n")
            handle.write(f"{self.top_of_model}\n")

            handle.write(f"{len(self.meteo_files)}\n")
            for meteo_file in self.meteo_files:
                directory = meteo_file.parent
                directory_str = str(directory)
                if not directory_str.endswith(os.sep):
                    directory_str += os.sep
                handle.write(directory_str + "\n")
                handle.write(meteo_file.name + "\n")

            handle.write(str(self.out_path) + "\n")
            handle.write(self.traj_file_name)
