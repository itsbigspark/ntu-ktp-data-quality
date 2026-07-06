"""
Command-line interface for DataQualify.

    dataqualify validate data.csv
    dataqualify validate data.csv --reference clean.csv --out issues.csv
    dataqualify validate data.csv --rules rules.json --format json

Designed to be dropped into a pipeline step (Airflow, cron, CI). Exit code is
non-zero when validation issues are found (unless --no-fail is given), so it can
gate a downstream job.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import pandas as pd


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        return pd.read_csv(path) if False else pd.read_parquet(path)
    return pd.read_csv(path)


def _cmd_validate(args: argparse.Namespace) -> int:
    import dataqualify as dq

    df = _read_table(args.input)
    rules = None
    if args.rules:
        with open(args.rules, "r", encoding="utf-8") as fh:
            rules = json.load(fh)
    reference = _read_table(args.reference) if args.reference else None

    issues = dq.validate(df, rules=rules, reference=reference,
                         typo_detection=not args.no_typo)

    if args.out:
        issues.to_csv(args.out, index=False)

    if args.format == "json":
        print(issues.to_json(orient="records"))
    elif args.format == "csv":
        print(issues.to_csv(index=False))
    else:  # summary
        n_cells = df.shape[0] * max(df.shape[1], 1)
        by_issue = issues["issue"].value_counts().to_dict() if len(issues) else {}
        print(f"rows: {df.shape[0]}  columns: {df.shape[1]}")
        print(f"issues: {len(issues)}  ({len(issues) / n_cells * 100:.2f}% of cells)")
        for k, v in by_issue.items():
            print(f"  {k}: {v}")
        if args.out:
            print(f"written: {args.out}")

    if len(issues) and not args.no_fail:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dataqualify",
        description="Explainable, weakly-supervised data quality for tabular data.",
    )
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    v = sub.add_parser("validate", help="validate a CSV/Parquet file")
    v.add_argument("input", help="path to the data file (.csv or .parquet)")
    v.add_argument("--reference", help="clean reference file for rule inference")
    v.add_argument("--rules", help="JSON file of explicit validation rules")
    v.add_argument("--out", help="write the issue report to this CSV path")
    v.add_argument("--format", choices=["summary", "json", "csv"], default="summary")
    v.add_argument("--no-typo", action="store_true", help="disable typo detection")
    v.add_argument("--no-fail", action="store_true",
                   help="always exit 0 even when issues are found")
    v.set_defaults(func=_cmd_validate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        import dataqualify as dq
        print(dq.__version__)
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
