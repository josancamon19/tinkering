#!/usr/bin/env python3
"""
Standalone script to compute and cache ALL stats combinations for OpenThoughts dataset.

Run this before using the Streamlit explorer to pre-cache all filter combinations:
    python -m tinkering.exploring_openthoughts.stats
"""

import argparse
import json
from itertools import product

from tinkering.exploring_openthoughts.common import (
    get_duckdb_connection,
    get_parquet_url,
    load_filter_options,
    build_where_clause,
    execute_with_retry,
    CACHE_DIR,
    HF_TOKEN,
)

# All stats are stored in this single organized file
ALL_STATS_FILE = CACHE_DIR / "all_stats.json"


def compute_stats_for_filters(con, parquet_url: str, filters: dict) -> dict:
    """Compute dataset statistics for given filters using existing DuckDB connection."""
    where_clause = build_where_clause(filters)

    basic_stats = execute_with_retry(
        con,
        f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT source) as unique_sources,
            COUNT(DISTINCT domain) as unique_domains,
            AVG(difficulty) as avg_difficulty,
            MIN(difficulty) as min_difficulty,
            MAX(difficulty) as max_difficulty
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
    """,
        "fetchone",
    )

    difficulty_dist = execute_with_retry(
        con,
        f"""
        SELECT difficulty, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY difficulty
        ORDER BY difficulty
    """,
        "fetchall",
    )

    source_dist = execute_with_retry(
        con,
        f"""
        SELECT source, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY source
        ORDER BY count DESC
    """,
        "fetchall",
    )

    domain_dist = execute_with_retry(
        con,
        f"""
        SELECT domain, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY domain
        ORDER BY count DESC
    """,
        "fetchall",
    )

    return {
        "total_rows": basic_stats[0],
        "unique_sources": basic_stats[1],
        "unique_domains": basic_stats[2],
        "avg_difficulty": basic_stats[3],
        "min_difficulty": basic_stats[4],
        "max_difficulty": basic_stats[5],
        "difficulty_distribution": {str(d): c for d, c in difficulty_dist},
        "source_distribution": {s: c for s, c in source_dist},
        "domain_distribution": {d: c for d, c in domain_dist},
    }


def load_all_stats() -> dict | None:
    """Load pre-computed stats from cache."""
    if ALL_STATS_FILE.exists():
        with open(ALL_STATS_FILE) as f:
            return json.load(f)
    return None


def save_all_stats(stats: dict):
    """Save all stats to cache file."""
    with open(ALL_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved to: {ALL_STATS_FILE}")


def get_stats_for_filters(all_stats: dict, source: str, domain: str) -> dict | None:
    """
    Look up pre-computed stats for given source/domain combination.

    Returns the most specific stats available for the filter combination.
    """
    if all_stats is None:
        return None

    # Generate lookup key
    key = f"{source}|{domain}"

    if key in all_stats.get("combinations", {}):
        return all_stats["combinations"][key]

    return None


def compute_all_combinations(force: bool = False) -> dict:
    """Compute stats for all source/domain combinations."""

    # Load filter options
    options = load_filter_options()
    if options is None:
        print("❌ Filter options not found! Run filters.py first.")
        return {}

    sources = ["all"] + options["sources"]
    domains = ["all"] + options["domains"]
    diff_min = options["difficulty_min"]
    diff_max = options["difficulty_max"]

    # Check existing cache
    existing = load_all_stats()
    if existing and not force:
        print(
            f"✓ Stats already cached with {len(existing.get('combinations', {}))} combinations."
        )
        print("  Use --force to recompute.")
        return existing

    print("=" * 60)
    print("Computing stats for ALL filter combinations")
    print("=" * 60)
    print(f"Sources: {len(sources)} (including 'all')")
    print(f"Domains: {len(domains)} (including 'all')")
    print(f"Difficulty range: {diff_min} - {diff_max}")
    print()

    # Generate all combinations
    combinations = list(product(sources, domains))
    total = len(combinations)
    print(f"Total combinations to compute: {total}")
    print()

    # Connect once and reuse
    print("Connecting to DuckDB...")
    con = get_duckdb_connection()
    parquet_url = get_parquet_url()

    all_stats = {
        "metadata": {
            "sources": sources,
            "domains": domains,
            "difficulty_min": diff_min,
            "difficulty_max": diff_max,
            "total_combinations": total,
        },
        "combinations": {},
    }

    for i, (source, domain) in enumerate(combinations, 1):
        key = f"{source}|{domain}"
        print(f"[{i}/{total}] Computing: source={source}, domain={domain}")

        # NOTE: Don't apply difficulty filters here - we want baseline stats
        # for each source/domain combo across ALL difficulties.
        # Difficulty filtering is for the UI, not for pre-computing stats.
        filters = {
            "source": source,
            "domain": domain,
        }

        try:
            stats = compute_stats_for_filters(con, parquet_url, filters)
            all_stats["combinations"][key] = stats
            print(f"         → {stats['total_rows']:,} rows")
        except Exception as e:
            print(f"         ❌ Error: {e}")
            all_stats["combinations"][key] = None

    con.close()

    # Save to cache
    print()
    save_all_stats(all_stats)

    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    base = all_stats["combinations"].get("all|all")
    if base:
        print(f"Total dataset: {base['total_rows']:,} rows")
        print()
        print("By domain:")
        for domain in options["domains"]:
            stats = all_stats["combinations"].get(f"all|{domain}")
            if stats:
                print(f"  {domain}: {stats['total_rows']:,} rows")
        print()
        print("By source:")
        for source in options["sources"]:
            stats = all_stats["combinations"].get(f"{source}|all")
            if stats:
                print(f"  {source}: {stats['total_rows']:,} rows")

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute all stats combinations for OpenThoughts dataset"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recompute even if cached"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show cached stats without computing"
    )
    args = parser.parse_args()

    if not HF_TOKEN:
        print("⚠️  Warning: No HF_TOKEN set. You may hit rate limits.")
        print()

    if args.show:
        stats = load_all_stats()
        if stats:
            print(f"Cached stats file: {ALL_STATS_FILE}")
            print(f"Total combinations: {len(stats.get('combinations', {}))}")
            print()
            print("Available combinations:")
            for key in sorted(stats.get("combinations", {}).keys()):
                s = stats["combinations"][key]
                if s:
                    print(f"  {key}: {s['total_rows']:,} rows")
        else:
            print("No cached stats found. Run without --show to compute.")
        return

    compute_all_combinations(force=args.force)


if __name__ == "__main__":
    main()
