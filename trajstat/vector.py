"""Light-weight vector layer abstraction used by the Python port."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class Field:
    name: str
    data_type: str
    length: int = 0
    decimal: int = 0


@dataclass
class AttributeTable:
    """Simplified attribute table backed by dictionaries."""

    fields: List[Field]
    records: List[Dict[str, Any]]

    def save(self, path: str | Path | None = None) -> None:
        if path is None:
            return
        import csv

        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[field.name for field in self.fields])
            writer.writeheader()
            for record in self.records:
                writer.writerow(record)

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return f"AttributeTable({len(self.records)} records)"


@dataclass
class VectorLayer:
    """Very small subset of the original MeteoInfo ``VectorLayer`` API."""

    layer_name: str
    shape_type: str
    fields: List[Field] = field(default_factory=list)
    shapes: List[Sequence[Any]] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    file_name: Path | None = None

    def edit_add_field(self, field_or_name: Field | str, data_type: str | None = None,
                       length: int = 0, decimal: int = 0) -> None:
        if isinstance(field_or_name, Field):
            field_obj = field_or_name
        else:
            if data_type is None:
                raise ValueError("data_type must be provided when adding a field by name")
            field_obj = Field(field_or_name, data_type, length, decimal)
        self.fields.append(field_obj)
        for record in self.records:
            record.setdefault(field_obj.name, None)

    def edit_insert_shape(self, shape: Sequence[Any], index: int) -> bool:
        self.shapes.insert(index, shape)
        self.records.insert(index, {})
        for field in self.fields:
            self.records[index].setdefault(field.name, None)
        return True

    def edit_cell_value(self, field: str | int, row: int, value: Any) -> None:
        field_name = self.fields[field].name if isinstance(field, int) else field
        self.records[row][field_name] = value

    def get_cell_value(self, field: str | int, row: int) -> Any:
        field_name = self.fields[field].name if isinstance(field, int) else field
        return self.records[row][field_name]

    def get_field_idx_by_name(self, name: str) -> int:
        for idx, field in enumerate(self.fields):
            if field.name == name:
                return idx
        return -1

    def get_field_number(self) -> int:
        return len(self.fields)

    def get_shape_num(self) -> int:
        return len(self.shapes)

    def get_shapes(self) -> List[Sequence[Any]]:
        return self.shapes

    def set_layer_name(self, name: str) -> None:
        self.layer_name = name

    def set_file_name(self, file_name: str | Path) -> None:
        self.file_name = Path(file_name)

    def save_file(self, file_name: str | Path | None = None) -> None:
        path = Path(file_name) if file_name else self.file_name
        if path is None:
            raise ValueError("A target path must be supplied")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for shape, record in zip(self.shapes, self.records):
            geometry = []
            for point in shape:
                if hasattr(point, "__dict__"):
                    geometry.append(dict(point.__dict__))
                else:
                    geometry.append(point)
            properties = {}
            for key, value in record.items():
                if hasattr(value, "isoformat"):
                    properties[key] = value.isoformat()
                else:
                    properties[key] = value
            data.append({"geometry": geometry, "properties": properties})
        import json

        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def get_attribute_table(self) -> AttributeTable:
        return AttributeTable(self.fields, self.records)
