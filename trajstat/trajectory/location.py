"""Location model for trajectory start points."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Location:
    """Geographic location expressed as latitude, longitude and height."""

    latitude: float
    longitude: float
    height: float

    @classmethod
    def from_string(cls, loc_str: str) -> "Location":
        """Create a :class:`Location` from a whitespace separated string.

        The Java implementation accepted values in the order
        ``latitude longitude height``.  The behaviour is preserved here so
        existing CONTROL files remain compatible.
        """

        parts = loc_str.strip().split()
        if len(parts) != 3:
            raise ValueError(
                "Location strings must contain latitude, longitude and height"
            )
        latitude, longitude, height = map(float, parts)
        return cls(latitude=latitude, longitude=longitude, height=height)

    def to_control_string(self) -> str:
        """Render the location as required by the CONTROL file."""

        return f"{self.latitude:.2f} {self.longitude:.2f} {self.height:.2f}"
