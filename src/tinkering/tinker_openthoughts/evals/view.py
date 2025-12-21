"""
Evaluation Results Viewer

A minimal Streamlit dashboard for viewing GPQA and LiveCodeBench evaluation results.

Usage:
    streamlit run src/tinkering/tinker_openthoughts/evals/view.py

For AIME results:
    Run `inspect view` to view AIME evaluation logs (stored as .eval files).
"""

import json
import streamlit as st
from pathlib import Path

LOGS_DIR = Path(__file__).parents[4] / "logs"


def find_result_files(eval_type: str) -> list[Path]:
    """Find all result files for a given eval type."""
    if eval_type == "gpqa":
        pattern = "**/gpqa_diamond_results.jsonl"
    else:
        pattern = "**/livecodebench_results.jsonl"
    return sorted(LOGS_DIR.glob(pattern), reverse=True)


def load_results(path: Path) -> list[dict]:
    """Load JSONL results file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_run_name(path: Path) -> str:
    """Extract a readable run name from path."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "openthoughts" and i + 2 < len(parts):
            return f"{parts[i+1][:40]}... / {parts[i+2]}"
    return str(path.parent.relative_to(LOGS_DIR))


def main():
    st.set_page_config(page_title="Eval Viewer", layout="wide")
    st.title("📊 Evaluation Results Viewer")

    # Sidebar
    st.sidebar.header("Settings")

    eval_type = st.sidebar.radio("Eval Type", ["gpqa", "livecodebench"])

    # Info about AIME
    st.sidebar.divider()
    st.sidebar.info("**AIME Results**\n\nRun `inspect view` to view AIME logs (.eval files)")

    # Find available result files
    result_files = find_result_files(eval_type)

    if not result_files:
        st.warning(f"No {eval_type} result files found in {LOGS_DIR}")
        return

    # Select run
    run_options = {get_run_name(p): p for p in result_files}
    selected_run = st.sidebar.selectbox("Run", list(run_options.keys()))
    results_path = run_options[selected_run]

    # Load results
    results = load_results(results_path)

    # Filter options
    st.sidebar.divider()
    filter_correct = st.sidebar.radio(
        "Filter", ["All", "Correct Only", "Incorrect Only"]
    )

    if filter_correct == "Correct Only":
        results = [r for r in results if r.get("is_correct")]
    elif filter_correct == "Incorrect Only":
        results = [r for r in results if not r.get("is_correct")]

    # Stats
    total = len(load_results(results_path))
    correct = sum(1 for r in load_results(results_path) if r.get("is_correct"))
    accuracy = correct / total if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Correct", correct)
    col3.metric("Accuracy", f"{accuracy:.1%}")

    st.divider()

    # Result navigation
    if not results:
        st.info("No results match the current filter.")
        return

    result_idx = st.slider(
        "Sample", 0, len(results) - 1, 0, format=f"Sample %d of {len(results)}"
    )
    result = results[result_idx]

    # Display result
    is_correct = result.get("is_correct", False)
    status = "✅ Correct" if is_correct else "❌ Incorrect"
    st.subheader(f"Sample {result['index']} — {status}")

    # GPQA specific fields
    if eval_type == "gpqa":
        cols = st.columns(2)
        cols[0].write(f"**Expected:** {result.get('correct_answer', 'N/A')}")
        cols[1].write(f"**Extracted:** {result.get('extracted_answer', 'N/A')}")

    # LiveCodeBench specific fields
    if eval_type == "livecodebench":
        cols = st.columns(2)
        cols[0].write(f"**Difficulty:** {result.get('difficulty', 'N/A')}")
        if result.get("error"):
            cols[1].write(f"**Error:** {result.get('error')}")

    # Prompt
    with st.expander("📝 Prompt", expanded=False):
        st.markdown(result.get("prompt", "No prompt"))

    # Response
    with st.expander("💬 Response", expanded=True):
        st.markdown(result.get("response", "No response"))


if __name__ == "__main__":
    main()

