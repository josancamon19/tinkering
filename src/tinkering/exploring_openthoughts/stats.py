#!/usr/bin/env python3
"""
Standalone script to compute and cache stats for OpenThoughts dataset.

Run this to precompute stats for specific filters:
    python -m tinkering._4c_compute_stats
    python -m tinkering._4c_compute_stats --domain code --difficulty-min 8
"""

import argparse

from tinkering.exploring_openthoughts.common import (
    get_duckdb_connection,
    get_parquet_url,
    load_filter_options,
    get_cache_key,
    load_cached_stats,
    save_stats_to_cache,
    build_where_clause,
    CACHE_DIR,
    HF_TOKEN,
)


def compute_stats(filters: dict) -> dict:
    """Compute dataset statistics using DuckDB."""
    print("Connecting to DuckDB...")
    con = get_duckdb_connection()
    parquet_url = get_parquet_url()
    where_clause = build_where_clause(filters)

    print(f"WHERE clause: {where_clause}")
    print()

    print("Computing basic stats...")
    count_query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT source) as unique_sources,
            COUNT(DISTINCT domain) as unique_domains,
            AVG(difficulty) as avg_difficulty,
            MIN(difficulty) as min_difficulty,
            MAX(difficulty) as max_difficulty
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
    """
    basic_stats = con.execute(count_query).fetchone()
    print(f"  Total rows: {basic_stats[0]:,}")

    print("Computing difficulty distribution...")
    difficulty_query = f"""
        SELECT difficulty, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY difficulty
        ORDER BY difficulty
    """
    difficulty_dist = con.execute(difficulty_query).fetchall()

    print("Computing source distribution...")
    source_query = f"""
        SELECT source, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY source
        ORDER BY count DESC
    """
    source_dist = con.execute(source_query).fetchall()

    print("Computing domain distribution...")
    domain_query = f"""
        SELECT domain, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY domain
        ORDER BY count DESC
    """
    domain_dist = con.execute(domain_query).fetchall()

    con.close()

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


def main():
    parser = argparse.ArgumentParser(description="Compute OpenThoughts dataset stats")
    parser.add_argument("--source", default="all", help="Filter by source")
    parser.add_argument("--domain", default="all", help="Filter by domain")
    parser.add_argument(
        "--difficulty-min", type=int, default=None, help="Min difficulty"
    )
    parser.add_argument(
        "--difficulty-max", type=int, default=None, help="Max difficulty"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recompute even if cached"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OpenThoughts Stats Computation")
    print("=" * 60)

    if not HF_TOKEN:
        print("⚠️  Warning: No HF_TOKEN set. You may hit rate limits.")
        print()

    # Load filter options to get defaults
    options = load_filter_options()
    if options:
        diff_min = (
            args.difficulty_min
            if args.difficulty_min is not None
            else options["difficulty_min"]
        )
        diff_max = (
            args.difficulty_max
            if args.difficulty_max is not None
            else options["difficulty_max"]
        )
    else:
        diff_min = args.difficulty_min if args.difficulty_min is not None else 1
        diff_max = args.difficulty_max if args.difficulty_max is not None else 10
        print(
            "⚠️  Filter options not cached. Run _4b_compute_filters.py first for accurate defaults."
        )
        print()

    filters = {
        "difficulty_min": diff_min,
        "difficulty_max": diff_max,
        "source": args.source,
        "domain": args.domain,
    }

    print(f"Filters: {filters}")
    print()

    cache_key = get_cache_key(filters)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    # Check cache
    if not args.force:
        cached = load_cached_stats(cache_key)
        if cached:
            print(f"✓ Stats already cached at: {cache_file}")
            print()
            print("Stats:")
            print(f"  Total rows: {cached['total_rows']:,}")
            print(f"  Avg difficulty: {cached['avg_difficulty']:.2f}")
            print(f"  Unique sources: {cached['unique_sources']}")
            print(f"  Unique domains: {cached['unique_domains']}")
            print()
            print("Use --force to recompute.")
            return

    print("Computing stats from remote parquet files...")
    print("This may take a few minutes.")
    print()

    stats = compute_stats(filters)
    save_stats_to_cache(cache_key, stats)

    print()
    print(f"✓ Saved to: {cache_file}")
    print()
    print("Stats:")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Avg difficulty: {stats['avg_difficulty']:.2f}")
    print(f"  Unique sources: {stats['unique_sources']}")
    print(f"  Unique domains: {stats['unique_domains']}")
    print()
    print("Difficulty distribution:")
    for d, c in stats["difficulty_distribution"].items():
        print(f"  {d}: {c:,}")


if __name__ == "__main__":
    main()
