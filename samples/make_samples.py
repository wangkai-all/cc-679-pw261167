"""Generate the demo / regression sample batches.

Batch 1 (samples/data) is the discovery batch; batch 2 (samples/next) is a
later upload that exercises reference gaps. The fixtures encode the cases the
brief calls out:

* ``same_name``: a column literally named ``product_id`` in both tables whose
  values do NOT contain each other -> name similarity is not evidence.
* ``both_singles_contained``: two columns each individually contained in the
  parent, but whose *pairs* are not -> singles are only a necessary condition.
* ``order_matters``: (store, sku) is contained; (sku, store) is not, because
  the parent pairs are ordered differently.
* ``parent_duplicates``: a parent with duplicate rows must not raise coverage.
* ``null_parent``: all-null / empty columns that must not become high
  confidence references.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def build(base: Path) -> None:
    data = base / "data"
    nxt = base / "next"

    # ---- parents ---------------------------------------------------------
    write_csv(
        data / "stores.csv",
        ["store_id", "city", "region"],
        [
            ["S01", "Shanghai", "East"],
            ["S02", "Hangzhou", "East"],
            ["S03", "Chengdu", "West"],
            ["S04", "Wuhan", "Central"],
            ["S01", "Shanghai", "East"],  # duplicate parent row: must not add coverage
        ],
    )
    write_csv(
        data / "products.csv",
        ["sku", "name", "category"],
        [
            ["SKU-100", "Notebook", "Stationery"],
            ["SKU-101", "Pen", "Stationery"],
            ["SKU-102", "Mug", "Home"],
            ["SKU-103", "Lamp", "Home"],
            ["SKU-104", "Cable", "Electronics"],
        ],
    )
    write_csv(
        data / "store_sku_catalog.csv",
        ["store_id", "sku", "shelf"],
        [
            ["S01", "SKU-100", "A1"],
            ["S01", "SKU-101", "A2"],
            ["S02", "SKU-100", "B1"],
            ["S02", "SKU-102", "B2"],
            ["S03", "SKU-103", "C1"],
            ["S04", "SKU-104", "D1"],
        ],
    )
    # A parent keyed the same way but with reversed column order semantics.
    write_csv(
        data / "sku_store_catalog.csv",
        ["sku", "store_id", "shelf"],
        [
            ["SKU-100", "S01", "A1"],
            ["SKU-101", "S01", "A2"],
            ["SKU-100", "S02", "B1"],
            ["SKU-102", "S02", "B2"],
        ],
    )
    # Same column NAME as a child column but disjoint values.
    write_csv(
        data / "legacy_products.csv",
        ["product_id", "label"],
        [
            ["P-9001", "Obsolete A"],
            ["P-9002", "Obsolete B"],
        ],
    )

    # ---- children --------------------------------------------------------
    write_csv(
        data / "orders.csv",
        ["order_id", "store_id", "sku", "qty", "ordered_at"],
        [
            ["O-1", "S01", "SKU-100", 2, "2026-01-04"],
            ["O-2", "S01", "SKU-101", 1, "2026-01-04"],
            ["O-3", "S02", "SKU-102", 5, "2026-01-05"],
            ["O-4", "S03", "SKU-103", 1, "2026-01-06"],
            ["O-5", "S01", "SKU-100", 3, "2026-01-07"],  # duplicate child key
            ["O-6", "", "SKU-101", 1, "2026-01-08"],       # null child key part
        ],
    )
    # Two single columns are each contained in their parent, but the PAIR is
    # not: (S02, SKU-103) and (S03, SKU-102) never co-occur in the catalog.
    write_csv(
        data / "order_lines.csv",
        ["line_id", "store_id", "sku", "amount"],
        [
            ["L-1", "S02", "SKU-103", 19.0],
            ["L-2", "S03", "SKU-102", 23.5],
            ["L-3", "S01", "SKU-100", 4.0],
            ["L-4", "S02", "SKU-100", 8.5],
        ],
    )
    # Composite child whose (store_id, sku) pairs are all real catalog pairs.
    write_csv(
        data / "inventory_snapshots.csv",
        ["snapshot_id", "store_id", "sku", "on_hand"],
        [
            ["INV-1", "S01", "SKU-100", 12],
            ["INV-2", "S02", "SKU-102", 4],
            ["INV-3", "S03", "SKU-103", 0],
            ["INV-4", "S01", "SKU-100", 11],
        ],
    )
    # A plain numeric reference to products by a differently-named column.
    write_csv(
        data / "returns.csv",
        ["return_id", "store_id", "item_code"],
        [
            ["R-1", "S01", "SKU-100"],
            ["R-2", "S03", "SKU-103"],
            ["R-3", "S02", "SKU-102"],
        ],
    )
    # Same column NAME and same type as legacy_products.product_id, but the
    # values are disjoint. Name similarity is not evidence.
    write_csv(
        data / "supplier_items.csv",
        ["item_id", "product_id", "supplier"],
        [
            ["SI-1", "SUP-1001", "Northwind"],
            ["SI-2", "SUP-1002", "Acme"],
        ],
    )
    # product_id values are NOT the same domain as products.sku.
    write_csv(
        data / "legacy_sales.csv",
        ["sale_id", "product_id", "store_id"],
        [
            ["LS-1", "P-9001", "S01"],
            ["LS-2", "P-9002", "S02"],
        ],
    )
    # Degenerate: an empty table and an all-null column.
    write_csv(data / "empty_feed.csv", ["store_id", "note"], [])
    write_csv(
        data / "null_feed.csv",
        ["feed_id", "store_id"],
        [["F-1", ""], ["F-2", ""], ["F-3", ""]],
    )
    # Integer-typed child referencing a string parent -> auto mapping refuses.
    write_csv(
        data / "kiosk_shifts.csv",
        ["shift_id", "store_no"],
        [[1, 1], [2, 2]],
    )
    write_csv(
        data / "store_numbers.csv",
        ["store_no"],
        [[1], [2], [3]],
    )
    write_csv(
        data / "store_numbers_text.csv",
        ["store_no"],
        [["1"], ["2"], ["3"]],
    )

    # ---- next batch ------------------------------------------------------
    write_csv(
        nxt / "stores.csv",
        ["store_id", "city", "region"],
        [["S01", "Shanghai", "East"], ["S02", "Hangzhou", "East"], ["S03", "Chengdu", "West"]],
    )
    write_csv(
        nxt / "products.csv",
        ["sku", "name", "category"],
        [["SKU-100", "Notebook", "Stationery"], ["SKU-102", "Mug", "Home"], ["SKU-103", "Lamp", "Home"]],
    )
    write_csv(
        nxt / "store_sku_catalog.csv",
        ["store_id", "sku", "shelf"],
        [["S01", "SKU-100", "A1"], ["S02", "SKU-102", "B2"], ["S03", "SKU-103", "C1"]],
    )
    # Two orphan lines: unknown store S99, and SKU-999 not in products.
    write_csv(
        nxt / "orders.csv",
        ["order_id", "store_id", "sku", "qty"],
        [
            ["O-11", "S01", "SKU-100", 1],
            ["O-12", "S99", "SKU-102", 2],
            ["O-13", "S02", "SKU-999", 4],
        ],
    )
    # Composite reference with a pair that is absent even though each single
    # value exists: S02 and SKU-103 are both in the catalog, just not together.
    write_csv(
        nxt / "inventory_snapshots.csv",
        ["snapshot_id", "store_id", "sku", "on_hand"],
        [
            ["INV-11", "S01", "SKU-100", 9],
            ["INV-12", "S02", "SKU-103", 2],
        ],
    )
    write_csv(
        nxt / "order_lines.csv",
        ["line_id", "store_id", "sku", "amount"],
        [["L-11", "S01", "SKU-100", 4.0], ["L-12", "S02", "SKU-999", 2.0]],
    )


def write_manifest_example(base: Path) -> None:
    relations = [
        {
            "status": "confirmed",
            "child_table": "orders",
            "child_cols": ["store_id"],
            "parent_table": "stores",
            "parent_cols": ["store_id"],
            "mapping_mode": "auto",
            "effective": ["none"],
            "child_logical": ["string"],
            "parent_logical": ["string"],
            "null_policy": "skip_child_nulls",
        },
        {
            "status": "confirmed",
            "child_table": "orders",
            "child_cols": ["sku"],
            "parent_table": "products",
            "parent_cols": ["sku"],
            "mapping_mode": "auto",
            "effective": ["none"],
            "child_logical": ["string"],
            "parent_logical": ["string"],
            "null_policy": "skip_child_nulls",
        },
        {
            "status": "confirmed",
            "child_table": "inventory_snapshots",
            "child_cols": ["store_id", "sku"],
            "parent_table": "store_sku_catalog",
            "parent_cols": ["store_id", "sku"],
            "mapping_mode": "auto",
            "effective": ["none", "none"],
            "child_logical": ["string", "string"],
            "parent_logical": ["string", "string"],
            "null_policy": "skip_child_nulls",
        },
        {
            "status": "excluded",
            "child_table": "legacy_sales",
            "child_cols": ["product_id"],
            "parent_table": "legacy_products",
            "parent_cols": ["product_id"],
            "mapping_mode": "auto",
            "effective": ["none"],
            "child_logical": ["string"],
            "parent_logical": ["string"],
            "null_policy": "skip_child_nulls",
        },
    ]
    for rel in relations:
        rel["candidate_key"] = (
            f"{rel['child_table']}[{','.join(rel['child_cols'])}] -> "
            f"{rel['parent_table']}[{','.join(rel['parent_cols'])}] @{rel['mapping_mode']}"
        )
    (base / "confirmed_relations.json").write_text(
        json.dumps({"dataset": "demo-batch-1", "relations": relations}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    build(here)
    write_manifest_example(here)
    print("samples written to", here)
