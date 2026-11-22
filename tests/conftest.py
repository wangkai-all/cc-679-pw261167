from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.fixture
def make_csv(tmp_path):
    def _make(name: str, header: list[str], rows: list[list[object]]) -> Path:
        return write_csv(tmp_path / f"{name}.csv", header, rows)

    return _make


@pytest.fixture
def samples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "samples"


