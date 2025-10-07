"""High level trajectory configuration utilities."""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from .control import TrajControl


@dataclass
class TrajConfig(TrajControl):
    """Extended configuration supporting multiple start hours and days."""

    start_hours: List[int] = field(default_factory=lambda: [6])
    start_day: int = 1
    end_day: int = 30
    traj_execute_file: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.traj_execute_file is None:
            self.traj_execute_file = Path("working") / "hyts_std.exe"
        else:
            self.traj_execute_file = Path(self.traj_execute_file)

    # ------------------------------------------------------------------
    def get_start_hours_string(self) -> str:
        return " ".join(f"{hour:2d}" for hour in self.start_hours).strip()

    def set_start_hours_from_string(self, value: str) -> None:
        self.start_hours = [int(part) for part in value.strip().split()]

    def update_start_end_days(self) -> None:
        self.start_day = 1
        _, self.end_day = calendar.monthrange(self.start_time.year, self.start_time.month)

    def load_control_file(self, file_name: str | Path) -> None:  # type: ignore[override]
        super().load_control_file(file_name)
        self.start_hours = [self.start_time.hour]
        self.update_start_end_days()

    # ------------------------------------------------------------------
    def get_start_hours_num(self) -> int:
        return len(self.start_hours)

    def get_day_num(self) -> int:
        return self.end_day - self.start_day + 1

    def get_time_num(self) -> int:
        return self.get_day_num() * self.get_start_hours_num()

    def update_start_time(self, day_index: int, hour_index: int) -> None:
        dt = self.start_time.replace(day=self.start_day + day_index, hour=self.start_hours[hour_index])
        self.start_time = dt
        self.traj_file_name = dt.strftime("%y%m%d%H")

    def init_start_time(self, year: int, month: int) -> None:
        self.start_time = datetime(year, month, self.start_day, self.start_hours[0], 0, 0)

    def get_traj_execute_file_name(self) -> str:
        return str(self.traj_execute_file)
