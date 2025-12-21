import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path

import chz
from datasets import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

from tinkering.exploring_openthoughts.common import (
    get_duckdb_connection,
    get_parquet_url,
    load_filter_options,
    build_where_clause,
    execute_with_retry,
    HF_TOKEN,
)

console = Console()


# ============================================================================
# Enums for CLI choices
# ============================================================================


class Domain(Enum):
    ALL = "all"
    CODE = "code"
    MATH = "math"
    SCIENCE = "science"


class Source(Enum):
    ALL = "all"
    OPENMATH = "ai2-adapt-dev/openmath-2-math"
    OPENCODE = "nvidia/OpenCodeReasoning"
    ORGANIC_CHEM = "organic-chemistry-questions"
    PHYSICS = "stackexchange-physics"
    CODEGOLF = "stackexchange_codegolf"


# ============================================================================
# Token Estimation
# ============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate tokens in text (approx len(chars) / 4)."""
    return len(text) // 4 if text else 0


def get_assistant_content(conversations: list[dict]) -> str:
    """Extract assistant message content from conversation."""
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        if role in ("gpt", "assistant"):
            return turn.get("value", turn.get("content", ""))
    return ""


def get_user_content(conversations: list[dict]) -> str:
    """Extract user/human message content from conversation."""
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        if role in ("human", "user"):
            return turn.get("value", turn.get("content", ""))
    return ""


def parse_thinking_and_output(assistant_content: str) -> tuple[str, str]:
    """
    Parse assistant content to extract thinking tokens and output.

    Returns:
        (thinking, output) tuple
    """
    # Match <think>...</think> pattern
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, assistant_content, re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        # Output is everything after </think>
        output = assistant_content[match.end() :].strip()
        return thinking, output
    else:
        # No thinking tags found, treat entire content as output
        return "", assistant_content.strip()


def get_total_conversation_tokens(conversations: list[dict]) -> int:
    """Estimate total tokens across all messages in conversation."""
    total = 0
    for turn in conversations:
        content = turn.get("value", turn.get("content", ""))
        total += estimate_tokens(content)
    return total


# ============================================================================
# Subset Generation
# ============================================================================


def generate_subset_name(
    source: str,
    domain: str,
    max_tokens: int,
    max_assistant_tokens: int | None,
    limit: int,
) -> str:
    """Generate a descriptive name for the subset."""
    # Clean source name for filename
    source_clean = (
        source.replace("/", "_").replace("-", "_") if source != "all" else "all_sources"
    )
    domain_clean = domain if domain != "all" else "all_domains"
    at_suffix = f"_at{max_assistant_tokens}" if max_assistant_tokens else ""
    return (
        f"openthoughts_{domain_clean}_{source_clean}_t{max_tokens}{at_suffix}_n{limit}"
    )


def fetch_filtered_data(
    source: str,
    domain: str,
    max_tokens: int,
    max_assistant_tokens: int | None,
    limit: int,
) -> list[dict]:
    """Fetch and filter data from OpenThoughts dataset using iterative batching."""

    console.print("[bold blue]Connecting to DuckDB...[/]")
    con = get_duckdb_connection()
    parquet_url = get_parquet_url()

    filters = {"source": source, "domain": domain}
    where_clause = build_where_clause(filters)

    # First get count
    console.print("[bold blue]Counting matching rows...[/]")
    count_query = f"""
        SELECT COUNT(*) 
        FROM read_parquet('{parquet_url}')
        WHERE {where_clause}
    """
    total_count = execute_with_retry(con, count_query, "fetchone")[0]
    console.print(f"  Found [green]{total_count:,}[/] rows matching filters")

    # Iterative fetching: keep fetching batches until we have enough
    BATCH_SIZE = 1000
    MAX_BATCHES = 100  # Safety limit: max 100k rows scanned
    filtered = []
    offset = 0
    total_fetched = 0
    total_filtered_out = 0

    console.print("[bold blue]Fetching and filtering data...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Collecting {limit} samples (0 so far)...", total=None
        )

        for batch_num in range(MAX_BATCHES):
            if len(filtered) >= limit:
                break

            if offset >= total_count:
                console.print(
                    f"  [yellow]Exhausted all {total_count:,} rows in dataset[/]"
                )
                break

            # Fetch next batch
            query = f"""
                SELECT *
                FROM read_parquet('{parquet_url}')
                WHERE {where_clause}
                LIMIT {BATCH_SIZE} OFFSET {offset}
            """

            progress.update(
                task,
                description=f"Batch {batch_num + 1}: fetching rows {offset:,}-{offset + BATCH_SIZE:,}...",
            )

            result = con.execute(query).fetchdf()
            batch_rows = result.to_dict("records")

            if not batch_rows:
                break

            total_fetched += len(batch_rows)

            # Process batch
            for ex in batch_rows:
                if len(filtered) >= limit:
                    break

                if isinstance(ex.get("conversations"), str):
                    ex["conversations"] = json.loads(ex["conversations"])

                convs = ex.get("conversations", [])

                # Check total conversation tokens
                total_tokens = get_total_conversation_tokens(convs)
                if total_tokens > max_tokens:
                    total_filtered_out += 1
                    continue

                # Check assistant message tokens (if limit specified)
                assistant_content = get_assistant_content(convs)
                assistant_tokens = estimate_tokens(assistant_content)
                if (
                    max_assistant_tokens is not None
                    and assistant_tokens > max_assistant_tokens
                ):
                    total_filtered_out += 1
                    continue

                # Add token metadata
                ex["_total_tokens"] = total_tokens
                ex["_assistant_tokens"] = assistant_tokens
                filtered.append(ex)

            progress.update(
                task,
                description=f"Collecting {limit} samples ({len(filtered)} so far)...",
            )

            offset += BATCH_SIZE

    con.close()

    # Summary
    console.print(f"  Scanned [blue]{total_fetched:,}[/] rows total")
    console.print(f"  Kept [green]{len(filtered):,}[/] rows after token filtering")
    console.print(f"  Filtered out [red]{total_filtered_out:,}[/] rows")

    if len(filtered) < limit:
        pass_rate = (len(filtered) / total_fetched * 100) if total_fetched > 0 else 0
        console.print(
            f"  [yellow]⚠️  Only found {len(filtered)}/{limit} samples "
            f"(pass rate: {pass_rate:.1f}%). Consider increasing max_tokens.[/]"
        )

    return filtered


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable types to Python native types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def create_sample_markdown(item: dict, index: int, samples_dir: Path) -> None:
    """Create a markdown file for a sample with separated sections."""
    convs = item.get("conversations", [])
    user_input = get_user_content(convs)
    assistant_content = get_assistant_content(convs)
    thinking, output = parse_thinking_and_output(assistant_content)

    # Build markdown content
    md_content = f"""# Sample {index + 1}

**Source:** {item.get("source", "unknown")}  
**Domain:** {item.get("domain", "unknown")}  
**Difficulty:** {item.get("difficulty", "N/A")}

---

## User Input

{user_input}

---

## Thinking Tokens

{thinking if thinking else "*No thinking tokens found*"}

---

## Output

{output}
"""

    sample_file = samples_dir / f"sample_{index + 1:02d}.md"
    with open(sample_file, "w") as f:
        f.write(md_content)


def save_subset(
    data: list[dict],
    subset_name: str,
    output_dir: Path,
    metadata: dict,
) -> Path:
    """Save subset to disk in HuggingFace format with metadata and samples."""
    subset_dir = output_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable_data = [convert_to_serializable(item) for item in data]

    # Save as HuggingFace Dataset
    console.print("[dim]  Creating HuggingFace Dataset...[/]")
    hf_dataset = Dataset.from_list(serializable_data)
    hf_dataset.save_to_disk(str(subset_dir / "dataset"))

    # Also save as JSONL for convenience
    # jsonl_file = subset_dir / "data.jsonl"
    # with open(jsonl_file, "w") as f:
    #     for item in serializable_data:
    #         f.write(json.dumps(item) + "\n")

    # Save metadata
    meta_file = subset_dir / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Create samples directory with 10 sample markdown files
    console.print("[dim]  Creating sample markdown files...[/]")
    samples_dir = subset_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(10, len(data))
    for i in range(num_samples):
        create_sample_markdown(data[i], i, samples_dir)

    console.print(f"[dim]  Created {num_samples} sample files in samples/[/]")

    return subset_dir


# ============================================================================
# CLI Config with chz
# ============================================================================


@chz.chz
class Config:
    """Configuration for OpenThoughts subset generation."""

    source: Source = Source.ALL
    """Difficulty source filter"""

    domain: Domain = Domain.ALL
    """Domain filter"""

    max_tokens: int = 8192
    """Maximum tokens for the entire conversation"""

    max_assistant_tokens: int | None = None
    """Maximum tokens for the assistant message (SFT target)"""

    limit: int = 100
    """Maximum number of items to extract"""

    output_dir: str = "./subsets"
    """Output directory for generated subsets"""

    list_sources: bool = False
    """List available sources and exit"""


def main(config: Config):
    # Header
    console.print(
        Panel.fit(
            "[bold magenta]🧠 OpenThoughts Subset Generator[/]\n"
            "[dim]Author: Joan Cabezas[/]",
            border_style="magenta",
        )
    )
    console.print()

    # Check for HF token
    if not HF_TOKEN:
        console.print(
            "[yellow]⚠️  Warning: No HF_TOKEN set. You may hit rate limits.[/]"
        )
        console.print()

    # Load filter options
    filter_options = load_filter_options()
    if filter_options is None:
        console.print("[red]❌ Filter options not found![/]")
        console.print(
            "Run: [cyan]python -m tinkering.exploring_openthoughts.filters[/]"
        )
        return

    # List sources mode
    if config.list_sources:
        table = Table(title="Available Sources")
        table.add_column("Source", style="cyan")
        table.add_column("Enum Value", style="dim")
        for src in Source:
            table.add_row(src.value, src.name)
        console.print(table)
        return

    # Extract values from enums
    source_val = config.source.value
    domain_val = config.domain.value

    # Validate source against filter options
    if source_val != "all" and source_val not in filter_options["sources"]:
        console.print(f"[red]❌ Unknown source: {source_val}[/]")
        console.print("Available sources:")
        for src in filter_options["sources"]:
            console.print(f"  - {src}")
        return

    # Show config
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Parameter", style="bold")
    config_table.add_column("Value", style="green")
    config_table.add_row("Source", source_val)
    config_table.add_row("Domain", domain_val)
    config_table.add_row("Max Tokens (conversation)", str(config.max_tokens))
    config_table.add_row("Max Tokens (assistant)", str(config.max_assistant_tokens))
    config_table.add_row("Limit", str(config.limit))
    config_table.add_row("Output Directory", config.output_dir)
    console.print(config_table)
    console.print()

    # Generate subset
    subset_name = generate_subset_name(
        source_val,
        domain_val,
        config.max_tokens,
        config.max_assistant_tokens,
        config.limit,
    )

    console.print(f"[bold]Subset name:[/] [cyan]{subset_name}[/]")
    console.print()

    # Fetch and filter data
    data = fetch_filtered_data(
        source=source_val,
        domain=domain_val,
        max_tokens=config.max_tokens,
        max_assistant_tokens=config.max_assistant_tokens,
        limit=config.limit,
    )

    if not data:
        console.print("[yellow]⚠️  No data matched the filters![/]")
        return

    # Build metadata
    metadata = {
        "name": subset_name,
        "author": "Joan Cabezas",
        "created_at": datetime.now().isoformat(),
        "base_dataset": "open-thoughts/OpenThoughts3-1.2M",
        "filters": {
            "source": source_val,
            "domain": domain_val,
            "max_tokens": config.max_tokens,
            "max_assistant_tokens": config.max_assistant_tokens,
            "limit": config.limit,
        },
        "stats": {
            "total_rows": len(data),
            "avg_total_tokens": sum(d["_total_tokens"] for d in data) / len(data)
            if data
            else 0,
            "avg_assistant_tokens": sum(d["_assistant_tokens"] for d in data)
            / len(data)
            if data
            else 0,
        },
    }

    # Save
    console.print()
    console.print("[bold blue]Saving subset...[/]")
    output_path = Path(config.output_dir)
    subset_dir = save_subset(data, subset_name, output_path, metadata)

    # Summary
    console.print()
    summary = Table(title="✅ Subset Generated Successfully", show_header=False)
    summary.add_column("", style="bold")
    summary.add_column("", style="green")
    summary.add_row("Output directory", str(subset_dir))
    summary.add_row("HuggingFace dataset", str(subset_dir / "dataset"))
    summary.add_row("JSONL file", str(subset_dir / "data.jsonl"))
    summary.add_row("Metadata file", str(subset_dir / "metadata.json"))
    summary.add_row("Samples directory", str(subset_dir / "samples"))
    summary.add_row("Total rows", f"{len(data):,}")
    summary.add_row(
        "Avg tokens/conversation", f"{metadata['stats']['avg_total_tokens']:.1f}"
    )
    summary.add_row(
        "Avg tokens/assistant", f"{metadata['stats']['avg_assistant_tokens']:.1f}"
    )
    console.print(summary)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
