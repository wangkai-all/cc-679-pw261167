"""Command-line ingress gate.

Exit codes: 0 = all confirmed references satisfied (publication allowed),
1 = at least one orphan/missing reference (publication refused),
2 = the manifest or inputs could not be read.
"""
from __future__ import annotations

import argparse
import json
import sys

from .core import ManifestError, load_manifest, validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="refgap-validate",
        description="对下一批文件执行已确认的引用完整性校验（仅约束 confirmed 关系）",
    )
    parser.add_argument("--manifest", required=True, help="已确认关系清单 JSON")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="新批次文件，可重复；NAME 必须与关系清单中的表名一致",
    )
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--json-out", help="将完整结果写入该路径")
    parser.add_argument("--label", default="", help="发布批次标签，写入结果")
    args = parser.parse_args(argv)

    files: dict[str, str] = {}
    for item in args.file:
        if "=" not in item:
            print(f"错误：--file 需要 NAME=PATH 形式，收到 {item!r}", file=sys.stderr)
            return 2
        name, path = item.split("=", 1)
        files[name.strip()] = path

    try:
        relations = load_manifest(args.manifest)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        print(f"错误：无法读取关系清单：{exc}", file=sys.stderr)
        return 2

    confirmed = [r for r in relations if r.get("status", "confirmed") == "confirmed"]
    if not confirmed:
        print("错误：清单中没有已确认的关系；不会用猜测代替约束", file=sys.stderr)
        return 2

    try:
        result = validate(confirmed, files, sample_limit=args.sample_limit, publication_label=args.label)
    except (OSError, ValueError, KeyError) as exc:
        # Missing/unreadable files fail closed with a distinct exit code; the
        # gate must never be mistaken for "publication refused because of
        # orphans" when it simply could not read the batch.
        print(f"错误：无法读取本批文件：{exc}", file=sys.stderr)
        return 2
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

    for rel in result["relations"]:
        mark = {"pass": "PASS", "fail": "FAIL", "not_applicable": "N/A", "error": "ERR"}[rel["status"]]
        print(f"[{mark}] {rel['candidate_key']}: {rel['reason']}")
        for item in rel["sample"]:
            print(f"        第 {item['row']} 行: {item['values']} ×{item['occurrences']}")
    print()
    if result["ok"]:
        print(f"发布允许：{result['enforced_count']} 条已确认引用全部满足。")
        return 0
    print(f"发布拒绝：{result['refusal_reason']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
