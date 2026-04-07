#!/usr/bin/env python3
"""
result_txt/pred/<run>/result.json 에서 group_averages, overall_average 만 모아 출력합니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize group_averages & overall_average from pred result.json files.")
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("result_txt/pred"),
        help="Directory containing run subfolders (default: result_txt/pred)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object per line (folder + extracted fields)",
    )
    args = parser.parse_args()

    root: Path = args.pred_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        raise SystemExit(1)

    run_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "result.json").is_file()
    )

    if not run_dirs:
        print(f"No result.json found under {root}", file=sys.stderr)
        raise SystemExit(0)

    # 모든 run에서 등장한 그룹 이름 순서 통일 (첫 파일 기준 + 나머지 합집합 정렬)
    group_names: set[str] = set()
    rows: list[dict] = []

    for d in run_dirs:
        path = d / "result.json"
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip] {path}: {e}", file=sys.stderr)
            continue

        ga = data.get("group_averages")
        oa = data.get("overall_average")
        if ga is not None and isinstance(ga, dict):
            group_names.update(ga.keys())
        rows.append(
            {
                "folder": d.name,
                "path": str(path),
                "group_averages": ga if isinstance(ga, dict) else {},
                "overall_average": oa,
            }
        )

    sorted_groups = sorted(group_names)

    if not rows:
        print("No valid result.json rows to print.", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        for r in rows:
            out = {
                "folder": r["folder"],
                "overall_average": r["overall_average"],
                "group_averages": r["group_averages"],
            }
            print(json.dumps(out, ensure_ascii=False))
        return

    # 표 형태: 헤더 + 각 run 한 줄
    col_w_folder = max(len("run"), max(len(r["folder"]) for r in rows))
    col_w_overall = 8

    header = f"{'run':<{col_w_folder}}  {'overall':>{col_w_overall}}"
    for g in sorted_groups:
        header += f"  {g:>12}"
    print(header)
    print("-" * len(header))

    for r in rows:
        oa = r["overall_average"]
        oa_s = f"{oa:.2f}" if isinstance(oa, (int, float)) else str(oa)
        line = f"{r['folder']:<{col_w_folder}}  {oa_s:>{col_w_overall}}"
        ga = r["group_averages"]
        for g in sorted_groups:
            v = ga.get(g) if isinstance(ga, dict) else None
            if v is None:
                line += f"  {'—':>12}"
            elif isinstance(v, (int, float)):
                line += f"  {v:>12.2f}"
            else:
                line += f"  {str(v):>12}"
        print(line)


if __name__ == "__main__":
    main()
