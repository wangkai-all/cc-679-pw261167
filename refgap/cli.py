"""CLI for discovery runs and export of confirmed relations."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .loader import LoadError, load_table
from .search import run_search, SearchParams, CONTAINED, PARTIAL
from .store import Store


def _parse_files(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--file 需要 NAME=PATH 形式，收到 {item!r}")
        name, path = item.split("=", 1)
        out.append((name.strip(), path))
    return out


def cmd_search(args: argparse.Namespace) -> int:
    store = Store(args.db)
    dataset_id = store.create_dataset(args.name)
    specs = []
    for name, path in _parse_files(args.file):
        table = load_table(name, path)
        specs.append(table)
        store.add_table(
            dataset_id,
            table.name,
            os.path.basename(path),
            table.path,
            table.source_hash,
            [{"name": c.name, "type": c.logical_type, "index": c.index} for c in table.columns],
            len(table.rows),
            table.warnings,
        )
    params = SearchParams(
        width_max=args.width,
        null_policy=args.nulls,
        mapping=args.mapping,
        max_verified=args.max_verified,
        max_seconds=args.max_seconds,
    )
    if args.columns:
        limits: dict[str, tuple[str, ...]] = {}
        for name, cols in (item.split(":", 1) for item in args.columns):
            limits[name] = tuple(c for c in cols.split(",") if c)
        params.column_limit = limits
    run_id = store.create_run(dataset_id, params.to_dict(), [{"name": t.name, "source_hash": t.source_hash} for t in specs])
    outcome = run_search(specs, params, on_result=lambda r: store.save_candidate(run_id, r))
    store.update_run_scope(run_id, outcome["scope"], "completed")
    print(json.dumps({"dataset_id": dataset_id, "run_id": run_id, "scope": outcome["scope"]}, ensure_ascii=False, indent=2))
    for res in outcome["results"]:
        if res.state in (CONTAINED, PARTIAL):
            print(
                f"{res.state:10s} {res.confidence:6s} {res.key}  "
                f"覆盖 {res.stats.get('key_coverage', 0):.1%} 缺失 {res.stats.get('missing_keys', 0)}"
            )
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    store = Store(args.db)
    run = store.get_run(args.run)
    if not run:
        raise SystemExit("找不到运行")
    cand = store.get_candidate(args.run, args.candidate)
    if not cand:
        raise SystemExit("找不到候选")
    spec = {
        "child_table": cand["child_table"],
        "child_cols": cand["child_cols"],
        "parent_table": cand["parent_table"],
        "parent_cols": cand["parent_cols"],
        "mapping_mode": cand["mapping_mode"],
        "effective": cand["effective"],
        "null_policy": cand["stats"].get("null_policy", "skip_child_nulls"),
    }
    store.set_decision(run["dataset_id"], args.candidate, args.action, spec, args.note or "")
    print(f"{args.action}: {args.candidate}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = Store(args.db)
    data = store.get_dataset(args.dataset)
    if not data:
        raise SystemExit("找不到数据集")
    relations = store.confirmed_relations(args.dataset)
    manifest = {"dataset": data["name"], "relations": relations}
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="refgap", description="跨表包含依赖发现")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("search", help="上传表并搜索候选引用")
    p.add_argument("--db", default=".refgap/store.sqlite3")
    p.add_argument("--name", default="dataset")
    p.add_argument("--file", action="append", default=[], help="NAME=PATH")
    p.add_argument("--width", type=int, default=2, choices=[1, 2])
    p.add_argument("--nulls", default="skip_child_nulls")
    p.add_argument("--mapping", default="auto")
    p.add_argument("--columns", action="append", default=[], help="表名:列1,列2 限定候选列")
    p.add_argument("--max-verified", type=int, default=400)
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.set_defaults(func=cmd_search)

    d = sub.add_parser("decide", help="确认或排除候选")
    d.add_argument("--db", default=".refgap/store.sqlite3")
    d.add_argument("--run", required=True)
    d.add_argument("--candidate", required=True)
    d.add_argument("--action", required=True, choices=["confirmed", "excluded"])
    d.add_argument("--note")
    d.set_defaults(func=cmd_decide)

    e = sub.add_parser("export", help="导出已确认关系清单")
    e.add_argument("--db", default=".refgap/store.sqlite3")
    e.add_argument("--dataset", required=True)
    e.add_argument("--out")
    e.set_defaults(func=cmd_export)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LoadError as exc:
        print(f"加载失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
