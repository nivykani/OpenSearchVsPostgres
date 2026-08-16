"""
Filters the Open Library bulk dumps down to just what this project needs.

Input files (download these yourself from https://openlibrary.org/developers/dumps):
  - ol_dump_authors_*.txt.gz
  - ol_dump_works_*.txt.gz

Each line of these dumps is TAB-separated:
  type  key  revision  last_modified  JSON

We only care about `key` and the JSON column.

Output: a CSV with columns id,title,subtitle,author_names,first_publish_year
  - author_names is a Postgres-array literal string, e.g. {"Jane Austen","Someone Else"}
  - first_publish_year is the first 4-digit number found in first_publish_date,
    kept only if it falls in [1000, 2050], else left blank (NULL on COPY)
  - the Open Library permalink (https://openlibrary.org/works/{id}) is fully
    derivable from id alone, so it is NOT stored — build it at query/render time.

Row count is controlled via --limit, not via required-field filtering, since
compounding multiple sparse-field requirements collapses yield very fast on
this dataset (see project notes).

Usage:
  python3 clean_openlibrary.py \
      --authors ol_dump_authors_2026-08-01.txt.gz \
      --works ol_dump_works_2026-08-01.txt.gz \
      --out works_seed.csv \
      --limit 50000
"""

import argparse
import csv
import gzip
import json
import re
import sys

YEAR_RE = re.compile(r"\d{4}")
MIN_YEAR, MAX_YEAR = 1000, 2050


def build_author_lookup(authors_path: str) -> dict:
    """Stream the Authors dump and build {author_key: name}."""
    lookup = {}
    with gzip.open(authors_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            key, json_col = parts[1], parts[4]
            try:
                record = json.loads(json_col)
            except json.JSONDecodeError:
                continue
            name = record.get("name")
            if name:
                lookup[key] = name
            if i % 1_000_000 == 0:
                print(f"  ...authors processed: {i:,}", file=sys.stderr)
    print(f"Built author lookup: {len(lookup):,} authors", file=sys.stderr)
    return lookup


def extract_author_keys(work: dict) -> list:
    """
    Works store authors as a list of author_role structs, but the real dump
    has at least three shapes in the wild:
      1. {"author": {"key": "/authors/OL123A"}, "type": {...}}   (typical)
      2. {"key": "/authors/OL123A"}                               (flat)
      3. "/authors/OL123A"                                        (bare string)
    Handle all three rather than assuming the documented shape is the only one.
    """
    keys = []
    for entry in work.get("authors", []) or []:
        if isinstance(entry, str):
            keys.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        author_ref = entry.get("author") or entry
        if isinstance(author_ref, str):
            keys.append(author_ref)
            continue
        if isinstance(author_ref, dict):
            key = author_ref.get("key")
            if key:
                keys.append(key)
    return keys


def extract_year(first_publish_date: str) -> str:
    if not first_publish_date:
        return ""
    match = YEAR_RE.search(first_publish_date)
    if not match:
        return ""
    year = int(match.group())
    if MIN_YEAR <= year <= MAX_YEAR:
        return str(year)
    return ""


def to_pg_array_literal(values: list) -> str:
    """
    Build a Postgres array literal for COPY, e.g. {"Jane Austen","O. Author"}.
    Escapes double quotes and backslashes inside each element.
    """
    if not values:
        return "{}"
    escaped = []
    for v in values:
        v = v.replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'"{v}"')
    return "{" + ",".join(escaped) + "}"


def clean_works(works_path: str, author_lookup: dict, out_path: str, limit: int | None):
    written = 0
    skipped_no_title = 0
    skipped_incomplete = 0

    with gzip.open(works_path, "rt", encoding="utf-8") as f_in, \
         open(out_path, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(["id", "title", "subtitle", "author_names", "first_publish_year"])

        for i, line in enumerate(f_in, 1):
            if limit and written >= limit:
                break

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            key, json_col = parts[1], parts[4]

            try:
                work = json.loads(json_col)
            except json.JSONDecodeError:
                continue

            title = work.get("title")
            if not title:
                skipped_no_title += 1
                continue

            subtitle = work.get("subtitle", "") or ""

            author_keys = extract_author_keys(work)
            author_names = [author_lookup[k] for k in author_keys if k in author_lookup]

            year = extract_year(work.get("first_publish_date", ""))

            # Per project scope: title and at least one resolved author are
            # required. Year is legitimately sparse/optional, so rows are
            # kept even when it's missing -- size is controlled via --limit.
            if not author_names:
                skipped_incomplete += 1
                continue

            work_id = key.split("/")[-1]  # "/works/OL45804W" -> "OL45804W"

            writer.writerow([
                work_id,
                title,
                subtitle,
                to_pg_array_literal(author_names),
                year,
            ])
            written += 1

            if i % 1_000_000 == 0:
                print(f"  ...works scanned: {i:,} | written: {written:,}", file=sys.stderr)

    print(
        f"Done. Wrote {written:,} rows. "
        f"Skipped {skipped_no_title:,} with no title, "
        f"{skipped_incomplete:,} with no resolved author.",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description="Clean/filter Open Library dumps for the search-comparison-demo project.")
    parser.add_argument("--authors", required=True, help="Path to ol_dump_authors_*.txt.gz")
    parser.add_argument("--works", required=True, help="Path to ol_dump_works_*.txt.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of output rows")
    args = parser.parse_args()

    print("Building author lookup (this reads the whole Authors dump into memory)...", file=sys.stderr)
    author_lookup = build_author_lookup(args.authors)

    print("Streaming Works dump and writing cleaned CSV...", file=sys.stderr)
    clean_works(args.works, author_lookup, args.out, args.limit)


if __name__ == "__main__":
    main()