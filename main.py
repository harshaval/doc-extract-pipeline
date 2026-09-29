#!/usr/bin/env python3
"""
CLI entrypoint.

Usage:
    python main.py extract path/to/file.pdf
    python main.py extract path/to/folder/          # batch, all *.pdf in folder
    python main.py review                            # interactive review queue
    python main.py stats                              # queue summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import pipeline, review_queue
from review_app.cli_review import run_review_session


def cmd_extract(args: argparse.Namespace) -> None:
    target = Path(args.path)
    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    if target.is_dir():
        outputs = pipeline.process_directory(str(target))
    else:
        outputs = [pipeline.process_document(str(target))]

    for out in outputs:
        flag = "🔶 NEEDS REVIEW" if out.needs_review else "✅ auto-approved"
        if out.is_critical:
            flag = "🔴 CRITICAL - " + flag
        print(f"{out.document_id}  {Path(out.source_path).name}  "
              f"confidence={out.overall_confidence:.2f}  {flag}")

    print(f"\n{len(outputs)} document(s) processed. "
          f"Results written to output/results/, raw text to output/raw_text/.")
    pending = review_queue.list_pending()
    if pending:
        print(f"{len(pending)} document(s) awaiting review — run `python main.py review`.")


def cmd_review(args: argparse.Namespace) -> None:
    run_review_session()


def cmd_stats(args: argparse.Namespace) -> None:
    print(json.dumps(review_queue.stats(), indent=2))


def main():
    parser = argparse.ArgumentParser(description="OCR + LLM document extraction pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Run extraction on a file or folder of PDFs")
    p_extract.add_argument("path", help="Path to a PDF file or a folder of PDFs")
    p_extract.set_defaults(func=cmd_extract)

    p_review = sub.add_parser("review", help="Interactively review low-confidence extractions")
    p_review.set_defaults(func=cmd_review)

    p_stats = sub.add_parser("stats", help="Show review queue counts by status")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
