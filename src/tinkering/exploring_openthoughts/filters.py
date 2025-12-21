#!/usr/bin/env python3
"""
Standalone script to compute and cache filter options for OpenThoughts dataset.

Run this before using the Streamlit explorer:
    python -m tinkering._4b_compute_filters
"""

from tinkering.exploring_openthoughts.common import (
    get_duckdb_connection,
    get_parquet_url,
    load_filter_options,
    save_filter_options,
    execute_with_retry,
    OPTIONS_CACHE_FILE,
    HF_TOKEN,
)


def compute_filter_options() -> dict:
    """Query the dataset to get all unique filter values."""
    print("Connecting to DuckDB...")
    con = get_duckdb_connection()
    parquet_url = get_parquet_url()

    print("Fetching unique sources...")
    sources = execute_with_retry(
        con,
        f"SELECT DISTINCT source FROM read_parquet('{parquet_url}')",
        "fetchall",
    )
    sources = sorted([s[0] for s in sources if s[0]])
    print(f"  Found {len(sources)} sources")

    print("Fetching unique domains...")
    domains = execute_with_retry(
        con,
        f"SELECT DISTINCT domain FROM read_parquet('{parquet_url}')",
        "fetchall",
    )
    domains = sorted([d[0] for d in domains if d[0]])
    print(f"  Found {len(domains)} domains")

    print("Fetching difficulty range...")
    diff_range = execute_with_retry(
        con,
        f"SELECT MIN(difficulty), MAX(difficulty) FROM read_parquet('{parquet_url}')",
        "fetchone",
    )
    print(f"  Difficulty range: {diff_range[0]} - {diff_range[1]}")

    con.close()

    return {
        "sources": sources,
        "domains": domains,
        "difficulty_min": diff_range[0],
        "difficulty_max": diff_range[1],
    }


def main():
    print("=" * 60)
    print("OpenThoughts Filter Options Computation")
    print("=" * 60)

    if not HF_TOKEN:
        print("⚠️  Warning: No HF_TOKEN set. You may hit rate limits.")
        print("   Set HF_TOKEN in your .env file or environment.")
        print()

    # Check if already cached
    cached = load_filter_options()
    if cached:
        print(f"✓ Filter options already cached at: {OPTIONS_CACHE_FILE}")
        print(f"  Sources: {len(cached['sources'])}")
        print(f"  Domains: {len(cached['domains'])}")
        print(f"  Difficulty: {cached['difficulty_min']} - {cached['difficulty_max']}")
        print()
        response = input("Recompute? [y/N]: ").strip().lower()
        if response != "y":
            print("Skipping.")
            return

    print()
    print("Computing filter options from remote parquet files...")
    print("This may take a few minutes on first run.")
    print()

    options = compute_filter_options()
    save_filter_options(options)

    print()
    print(f"✓ Saved to: {OPTIONS_CACHE_FILE}")
    print()
    print("Filter options:")
    print(f"  Sources ({len(options['sources'])}):")
    for s in options["sources"][:10]:
        print(f"    - {s}")
    if len(options["sources"]) > 10:
        print(f"    ... and {len(options['sources']) - 10} more")
    print(f"  Domains ({len(options['domains'])}): {options['domains']}")
    print(f"  Difficulty: {options['difficulty_min']} - {options['difficulty_max']}")


if __name__ == "__main__":
    main()
