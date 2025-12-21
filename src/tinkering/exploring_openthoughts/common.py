"""
Shared configuration and utilities for OpenThoughts dataset exploration.
"""

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import duckdb
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

DATASET_NAME = "open-thoughts/OpenThoughts3-1.2M"
CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "openthoughts_stats"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OPTIONS_CACHE_FILE = CACHE_DIR / "filter_options.json"

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_parquet_url() -> str:
    """Get the parquet URL pattern for DuckDB."""
    return f"hf://datasets/{DATASET_NAME}/data/train-*.parquet"


def get_duckdb_connection():
    """Create a DuckDB connection configured for HuggingFace access."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


# ============================================================================
# Filter Options
# ============================================================================


def load_filter_options() -> dict | None:
    """Load cached filter options."""
    if OPTIONS_CACHE_FILE.exists():
        with open(OPTIONS_CACHE_FILE) as f:
            return json.load(f)
    return None


def save_filter_options(options: dict):
    """Save filter options to cache."""
    with open(OPTIONS_CACHE_FILE, "w") as f:
        json.dump(options, f, indent=2)


# ============================================================================
# Stats Cache
# ============================================================================


def get_cache_key(filters: dict[str, Any]) -> str:
    """Generate a cache key from filters."""
    filter_str = json.dumps(filters, sort_keys=True)
    return hashlib.md5(filter_str.encode()).hexdigest()


def load_cached_stats(cache_key: str) -> dict | None:
    """Load stats from cache if available."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def save_stats_to_cache(cache_key: str, stats: dict):
    """Save stats to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump(stats, f, indent=2)


# ============================================================================
# Query Helpers
# ============================================================================


def build_where_clause(filters: dict[str, Any]) -> str:
    """Build SQL WHERE clause from filters."""
    conditions = []
    if filters.get("difficulty_min") is not None:
        conditions.append(f"difficulty >= {filters['difficulty_min']}")
    if filters.get("difficulty_max") is not None:
        conditions.append(f"difficulty <= {filters['difficulty_max']}")
    if filters.get("source") and filters["source"] != "all":
        conditions.append(f"source = '{filters['source']}'")
    if filters.get("domain") and filters["domain"] != "all":
        conditions.append(f"domain = '{filters['domain']}'")

    return " AND ".join(conditions) if conditions else "1=1"


# ============================================================================
# Query Execution with Retry
# ============================================================================

RETRY_DELAY_SECONDS = 61


def execute_with_retry(
    con, query: str, fetch_method: str = "fetchall", max_retries: int = 5
):
    """Execute a query with retry logic on failure (handles HuggingFace rate limits)."""
    attempts = 0
    while True:
        try:
            result = con.execute(query)
            if fetch_method == "fetchall":
                return result.fetchall()
            elif fetch_method == "fetchone":
                return result.fetchone()
            else:
                return result
        except Exception as e:
            attempts += 1
            if max_retries and attempts >= max_retries:
                print(f"  ❌ Query failed after {attempts} attempts: {e}")
                raise
            print(f"  ⚠️  Query failed: {e}")
            print(f"  Retrying in {RETRY_DELAY_SECONDS} seconds... (attempt {attempts})")
            time.sleep(RETRY_DELAY_SECONDS)
