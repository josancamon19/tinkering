"""
Streamlit dashboard to explore OpenThoughts3-1.2M dataset.

Prerequisites:
    python -m tinkering._4b_compute_filters  # Run first to cache filter options
    streamlit run src/tinkering/_4_exploring_openthoughts.py
"""

import json
import re

import streamlit as st
import pandas as pd

from tinkering.exploring_openthoughts.common import (
    get_duckdb_connection,
    get_parquet_url,
    load_filter_options,
    get_cache_key,
    load_cached_stats,
    save_stats_to_cache,
    build_where_clause,
    HF_TOKEN,
)


# ============================================================================
# Data Fetching
# ============================================================================


def fetch_page(filters: dict, limit: int = 50, offset: int = 0) -> list[dict]:
    """Fetch a page of examples using DuckDB."""
    con = get_duckdb_connection()
    parquet_url = get_parquet_url()
    where_clause = build_where_clause(filters)

    query = f"""
        SELECT *
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        LIMIT {limit} OFFSET {offset}
    """

    result = con.execute(query).fetchdf()
    con.close()

    examples = result.to_dict("records")
    for ex in examples:
        if isinstance(ex.get("conversations"), str):
            ex["conversations"] = json.loads(ex["conversations"])
    return examples


def compute_stats(filters: dict) -> dict:
    """Compute dataset statistics using DuckDB."""
    con = get_duckdb_connection()
    parquet_url = get_parquet_url()
    where_clause = build_where_clause(filters)

    basic_stats = con.execute(f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT source) as unique_sources,
            COUNT(DISTINCT domain) as unique_domains,
            AVG(difficulty) as avg_difficulty,
            MIN(difficulty) as min_difficulty,
            MAX(difficulty) as max_difficulty
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
    """).fetchone()

    difficulty_dist = con.execute(f"""
        SELECT difficulty, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY difficulty ORDER BY difficulty
    """).fetchall()

    source_dist = con.execute(f"""
        SELECT source, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY source ORDER BY count DESC
    """).fetchall()

    domain_dist = con.execute(f"""
        SELECT domain, COUNT(*) as count
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
        GROUP BY domain ORDER BY count DESC
    """).fetchall()

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


# ============================================================================
# Conversation Rendering
# ============================================================================


def extract_thinking_and_response(content: str) -> tuple[str | None, str]:
    """Extract thinking trace and response from content."""
    patterns = [
        (r"<think>(.*?)</think>", re.DOTALL),
        (r"<\|thinking\|>(.*?)<\|/thinking\|>", re.DOTALL),
        (r"<thinking>(.*?)</thinking>", re.DOTALL),
    ]
    for pattern, flags in patterns:
        match = re.search(pattern, content, flags)
        if match:
            thinking = match.group(1).strip()
            response = re.sub(pattern, "", content, flags=flags).strip()
            return thinking, response
    return None, content


def get_conversation_preview(conversations, max_len: int = 100) -> str:
    """Get a short preview of the conversation."""
    if conversations is None or (
        hasattr(conversations, "__len__") and len(conversations) == 0
    ):
        return "No conversation"
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        if role in ("human", "user"):
            content = turn.get("value", turn.get("content", ""))
            return content[:max_len] + "..." if len(content) > max_len else content
    return "No user message found"


def estimate_conversation_tokens(conversations) -> int:
    """Estimate total tokens in a conversation (approx len(chars) / 4)."""
    if conversations is None or (
        hasattr(conversations, "__len__") and len(conversations) == 0
    ):
        return 0
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()
    total_chars = 0
    for turn in conversations:
        content = turn.get("value", turn.get("content", ""))
        if content:
            total_chars += len(content)
    return total_chars // 4


def render_conversation_detail(conversation):
    """Render a conversation with thinking trace separation."""
    if hasattr(conversation, "tolist"):
        conversation = conversation.tolist()
    if not conversation:
        st.warning("No conversation data")
        return

    for turn in conversation:
        role = turn.get("from", turn.get("role", "unknown"))
        content = turn.get("value", turn.get("content", ""))

        if role in ("human", "user"):
            st.markdown("### 🧑 User Question")
            st.markdown(content)
            st.divider()

        elif role in ("gpt", "assistant"):
            st.markdown("### 🤖 Assistant Response")
            thinking, response = extract_thinking_and_response(content)

            if thinking:
                with st.expander("💭 Thinking Trace", expanded=False):
                    st.code(thinking, language=None)
                st.markdown("#### 📝 Final Response")
                st.markdown(response)
            else:
                if len(content) > 5000:
                    with st.expander("View full response (long)", expanded=True):
                        st.markdown(content)
                else:
                    st.markdown(content)
            st.divider()


# ============================================================================
# Streamlit UI
# ============================================================================


def main():
    st.set_page_config(
        page_title="OpenThoughts3 Explorer", page_icon="🧠", layout="wide"
    )

    # Check for filter options first
    filter_options = load_filter_options()
    if filter_options is None:
        st.error("❌ Filter options not found!")
        st.markdown("""
        Run the filter computation script first:
        ```bash
        python -m tinkering._4b_compute_filters
        ```
        """)
        st.stop()

    st.title("🧠 OpenThoughts3-1.2M Explorer")

    # Sidebar
    st.sidebar.header("🔍 Filters")

    if not HF_TOKEN:
        st.sidebar.warning("⚠️ No HF_TOKEN set. May hit rate limits.")

    st.sidebar.success(
        f"✅ {len(filter_options['sources'])} sources, {len(filter_options['domains'])} domains"
    )

    difficulty_range = st.sidebar.slider(
        "Difficulty Range",
        min_value=filter_options["difficulty_min"],
        max_value=filter_options["difficulty_max"],
        value=(filter_options["difficulty_min"], filter_options["difficulty_max"]),
    )

    source = st.sidebar.selectbox("Source", ["all"] + filter_options["sources"])
    domain = st.sidebar.selectbox("Domain", ["all"] + filter_options["domains"])

    filters = {
        "difficulty_min": difficulty_range[0],
        "difficulty_max": difficulty_range[1],
        "source": source,
        "domain": domain,
    }

    # Tabs
    tab_table, tab_stats = st.tabs(["📊 Data Table", "📈 Statistics"])

    # Data Table Tab
    with tab_table:
        PAGE_SIZE = 50

        # Initialize pagination state
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        if "filters_hash" not in st.session_state:
            st.session_state.filters_hash = None

        # Reset to page 1 if filters changed
        current_filters_hash = str(filters)
        if st.session_state.filters_hash != current_filters_hash:
            st.session_state.current_page = 1
            st.session_state.filters_hash = current_filters_hash
            st.session_state.pop("examples", None)

        page = st.session_state.current_page
        offset = (page - 1) * PAGE_SIZE

        # Pagination controls
        col_prev, col_info, col_next, col_jump = st.columns([1, 2, 1, 2])

        with col_prev:
            if st.button("← Previous", disabled=(page <= 1), use_container_width=True):
                st.session_state.current_page -= 1
                st.session_state.pop("examples", None)
                st.rerun()

        with col_info:
            st.markdown(
                f"<div style='text-align: center; padding: 0.5rem;'><strong>Page {page}</strong></div>",
                unsafe_allow_html=True,
            )

        with col_next:
            # Disable next if we got fewer results than PAGE_SIZE
            examples_count = len(st.session_state.get("examples", []))
            at_end = examples_count > 0 and examples_count < PAGE_SIZE
            if st.button("Next →", disabled=at_end, use_container_width=True):
                st.session_state.current_page += 1
                st.session_state.pop("examples", None)
                st.rerun()

        with col_jump:
            jump_page = st.number_input(
                "Go to page",
                min_value=1,
                value=page,
                step=1,
                label_visibility="collapsed",
                key="jump_page_input",
            )
            if jump_page != page:
                st.session_state.current_page = jump_page
                st.session_state.pop("examples", None)
                st.rerun()

        # Auto-load data if not cached
        if "examples" not in st.session_state:
            with st.spinner("Loading..."):
                try:
                    st.session_state.examples = fetch_page(
                        filters, limit=PAGE_SIZE, offset=offset
                    )
                    st.session_state.current_offset = offset
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.examples = []

        examples = st.session_state.get("examples", [])

        if examples:
            st.caption(
                f"Showing {len(examples)} rows (offset {st.session_state.get('current_offset', 0)})"
            )

            table_data = [
                {
                    "idx": i,
                    "difficulty": ex.get("difficulty"),
                    "source": ex.get("source", "N/A"),
                    "domain": ex.get("domain", "N/A"),
                    "tokens": estimate_conversation_tokens(ex.get("conversations", [])),
                    "preview": get_conversation_preview(
                        ex.get("conversations", []), max_len=80
                    ),
                }
                for i, ex in enumerate(examples)
            ]

            df = pd.DataFrame(table_data)
            st.dataframe(
                df[["difficulty", "source", "domain", "tokens", "preview"]],
                use_container_width=True,
                hide_index=False,
            )

            st.divider()
            st.markdown("### 🔍 View Conversation")

            options = [
                f"{i}: {row['preview'][:50]}..." for i, row in enumerate(table_data)
            ]
            selected = st.selectbox("Select row", options, index=0)
            selected_idx = int(selected.split(":")[0])
            example = examples[selected_idx]

            col1, col2, col3 = st.columns(3)
            col1.metric("Difficulty", example.get("difficulty"))
            col2.metric("Source", example.get("source", "N/A"))
            col3.metric("Domain", example.get("domain", "N/A"))

            st.divider()
            render_conversation_detail(example.get("conversations", []))
        else:
            st.info("No results found for the current filters.")

    # Stats Tab
    with tab_stats:
        st.info(
            "Stats use DuckDB to query remote parquet files. Results are cached locally."
        )

        if st.button("📊 Compute Statistics", type="primary"):
            cache_key = get_cache_key(filters)
            cached = load_cached_stats(cache_key)

            if cached:
                st.success("✅ Loaded from cache!")
                stats = cached
            else:
                with st.spinner("Computing..."):
                    try:
                        stats = compute_stats(filters)
                        save_stats_to_cache(cache_key, stats)
                        st.success("✅ Computed and cached!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        stats = None

            if stats:
                st.session_state.current_stats = stats

        stats = st.session_state.get("current_stats")
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{stats['total_rows']:,}")
            col2.metric("Avg Difficulty", f"{stats['avg_difficulty']:.2f}")
            col3.metric("Unique Sources", stats["unique_sources"])
            col4.metric("Unique Domains", stats["unique_domains"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Difficulty Distribution**")
                diff_df = pd.DataFrame(
                    list(stats["difficulty_distribution"].items()),
                    columns=["Difficulty", "Count"],
                )
                st.bar_chart(diff_df.set_index("Difficulty"))
            with col2:
                st.markdown("**Source Distribution**")
                source_df = pd.DataFrame(
                    list(stats["source_distribution"].items()),
                    columns=["Source", "Count"],
                )
                st.bar_chart(source_df.set_index("Source"))

            st.markdown("**Domain Distribution**")
            domain_df = pd.DataFrame(
                list(stats["domain_distribution"].items()), columns=["Domain", "Count"]
            )
            st.bar_chart(domain_df.set_index("Domain"))


if __name__ == "__main__":
    main()
