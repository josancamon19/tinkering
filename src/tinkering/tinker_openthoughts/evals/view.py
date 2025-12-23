"""
Evaluation Results Viewer

A minimal Streamlit dashboard for viewing GPQA and LiveCodeBench evaluation results.
Organized by training run → step → eval type for tracing evolution.

Usage:
    streamlit run src/tinkering/tinker_openthoughts/evals/view.py

For AIME results:
    Run `inspect view` to view AIME evaluation logs (stored as .eval files).
"""

import json
import re
import streamlit as st
from pathlib import Path
from dataclasses import dataclass

LOGS_DIR = Path(__file__).parents[4] / "logs" / "openthoughts"


@dataclass
class RunParams:
    """Parsed training run parameters from config name."""

    dataset_name: str = ""
    train_split: float = 0.0
    batch_size: int = 0
    learning_rate: float = 0.0
    epochs: int = 0
    model_name: str = ""

    @classmethod
    def from_config_name(cls, config_name: str) -> "RunParams":
        """Parse run parameters from config name."""
        params = cls()

        # Parse config_name: {dataset}_s{split}_bs{batch}_lr{lr}_e{epochs}_{model}
        # Example: openthoughts_code_all_sources_t4096_n1000_s0.9_bs32_lr1e-05_e13_Qwen_Qwen3-4B-Instruct-2507
        match = re.match(
            r"(.+?)_s(\d+\.?\d*)_bs(\d+)_lr([^_]+)_e(\d+)_(.+)", config_name
        )
        if match:
            params.dataset_name = match.group(1)
            params.train_split = float(match.group(2))
            params.batch_size = int(match.group(3))
            params.learning_rate = float(match.group(4))
            params.epochs = int(match.group(5))
            params.model_name = match.group(6).replace("_", "/", 1)

        return params


def find_training_runs() -> dict[str, Path]:
    """Find all training runs (config_name/timestamp directories)."""
    runs = {}
    if not LOGS_DIR.exists():
        return runs

    for config_dir in LOGS_DIR.iterdir():
        if not config_dir.is_dir():
            continue
        for timestamp_dir in config_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue
            # Create readable name
            params = RunParams.from_config_name(config_dir.name)
            ds_short = params.dataset_name.replace("openthoughts_code_all_sources_", "")
            run_name = f"{ds_short} / {timestamp_dir.name}"
            runs[run_name] = timestamp_dir

    return dict(sorted(runs.items(), reverse=True))


def find_steps_in_run(run_path: Path) -> dict[int, dict[str, Path]]:
    """
    Find all steps and their eval results in a training run.
    Returns: {step: {"gpqa": path, "livecodebench": path}}
    """
    steps: dict[int, dict[str, Path]] = {}

    # Check each eval type directory
    for eval_type in ["gpqa", "livecodebench"]:
        eval_dir = run_path / eval_type
        if not eval_dir.exists():
            continue

        # Check for step subdirectories
        for item in eval_dir.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                try:
                    step = int(item.name.replace("step_", ""))
                    if step not in steps:
                        steps[step] = {}

                    # Find result file
                    result_file = (
                        item / "gpqa_diamond_results.jsonl"
                        if eval_type == "gpqa"
                        else item / "livecodebench_results.jsonl"
                    )
                    if result_file.exists():
                        steps[step][eval_type] = result_file
                except ValueError:
                    pass

        # Also check for results directly in eval_dir (legacy format without step)
        result_file = (
            eval_dir / "gpqa_diamond_results.jsonl"
            if eval_type == "gpqa"
            else eval_dir / "livecodebench_results.jsonl"
        )
        if result_file.exists():
            if -1 not in steps:  # Use -1 for "unknown step"
                steps[-1] = {}
            steps[-1][eval_type] = result_file

    return dict(sorted(steps.items()))


def load_results(path: Path) -> list[dict]:
    """Load JSONL results file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_previous_step(steps: dict[int, dict[str, Path]], current_step: int) -> int | None:
    """Get the previous step number, or None if no previous step."""
    sorted_steps = sorted(steps.keys())
    try:
        idx = sorted_steps.index(current_step)
        if idx > 0:
            return sorted_steps[idx - 1]
    except ValueError:
        pass
    return None


def compute_step_diff(
    current_results: list[dict], prev_results: list[dict]
) -> dict[int, str]:
    """
    Compare current results to previous step.
    Returns: {index: "new_pass" | "new_fail" | "same"}
    """
    prev_by_idx = {r["index"]: r.get("is_correct", False) for r in prev_results}
    diff = {}
    for r in current_results:
        idx = r["index"]
        curr_correct = r.get("is_correct", False)
        prev_correct = prev_by_idx.get(idx)

        if prev_correct is None:
            diff[idx] = "new"  # New sample not in previous
        elif curr_correct and not prev_correct:
            diff[idx] = "new_pass"  # Now solved
        elif not curr_correct and prev_correct:
            diff[idx] = "new_fail"  # Now failing
        else:
            diff[idx] = "same"
    return diff


def truncate_prompt(prompt: str, max_len: int = 80) -> str:
    """Truncate prompt for table display."""
    prompt = prompt.replace("\n", " ").strip()
    if len(prompt) > max_len:
        return prompt[:max_len] + "..."
    return prompt


def main():
    st.set_page_config(page_title="Eval Viewer", layout="wide")
    st.title("📊 Evaluation Results Viewer")

    # Sidebar - Training Run Selection
    st.sidebar.header("1. Select Training Run")

    training_runs = find_training_runs()

    if not training_runs:
        st.warning(f"No training runs found in {LOGS_DIR}")
        st.info("Run training first to generate evaluation logs.")
        return

    selected_run_name = st.sidebar.selectbox("Training Run", list(training_runs.keys()))
    run_path = training_runs[selected_run_name]

    # Parse and display run parameters
    config_name = run_path.parent.name
    params = RunParams.from_config_name(config_name)

    with st.sidebar.expander("Run Parameters", expanded=False):
        if params.model_name:
            st.text(f"Model: {params.model_name}")
        if params.batch_size:
            st.text(f"Batch Size: {params.batch_size}")
        if params.learning_rate:
            st.text(f"LR: {params.learning_rate}")
        if params.epochs:
            st.text(f"Epochs: {params.epochs}")
        if params.train_split:
            st.text(f"Train Split: {params.train_split}")

    # Find available steps
    st.sidebar.divider()
    st.sidebar.header("2. Select Step")

    steps = find_steps_in_run(run_path)

    if not steps:
        st.warning("No evaluation results found for this training run.")
        return

    # Create step options with available evals
    step_options = {}
    for step, evals in steps.items():
        eval_names = ", ".join(evals.keys())
        if step == -1:
            step_options[f"(no step) — {eval_names}"] = step
        else:
            step_options[f"Step {step} — {eval_names}"] = step

    selected_step_name = st.sidebar.selectbox("Step", list(step_options.keys()))
    selected_step = step_options[selected_step_name]
    available_evals = steps[selected_step]

    # Select eval type
    st.sidebar.divider()
    st.sidebar.header("3. Select Eval Type")

    eval_type = st.sidebar.radio("Eval Type", list(available_evals.keys()))

    # Info about AIME
    st.sidebar.divider()
    st.sidebar.info(
        "**AIME Results**\n\nRun `inspect view` to view AIME logs (.eval files)"
    )

    # Load results
    results_path = available_evals[eval_type]
    all_results = load_results(results_path)

    # Load previous step results for comparison
    prev_step = get_previous_step(steps, selected_step)
    prev_results = []
    step_diff = {}
    if prev_step is not None and eval_type in steps.get(prev_step, {}):
        prev_results = load_results(steps[prev_step][eval_type])
        step_diff = compute_step_diff(all_results, prev_results)

    # Filter options
    st.sidebar.divider()
    filter_option = st.sidebar.radio(
        "Filter",
        ["All", "Correct Only", "Incorrect Only", "🆕 Newly Solved", "🔻 Newly Failed"],
    )

    results = all_results
    if filter_option == "Correct Only":
        results = [r for r in results if r.get("is_correct")]
    elif filter_option == "Incorrect Only":
        results = [r for r in results if not r.get("is_correct")]
    elif filter_option == "🆕 Newly Solved":
        results = [r for r in results if step_diff.get(r["index"]) == "new_pass"]
    elif filter_option == "🔻 Newly Failed":
        results = [r for r in results if step_diff.get(r["index"]) == "new_fail"]

    # Main content - Stats header
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("is_correct"))
    accuracy = correct / total if total > 0 else 0

    # Compute improvements
    newly_solved = sum(1 for d in step_diff.values() if d == "new_pass")
    newly_failed = sum(1 for d in step_diff.values() if d == "new_fail")

    # Header with run info
    step_label = f"Step {selected_step}" if selected_step >= 0 else "No step info"
    st.subheader(f"{eval_type.upper()} @ {step_label}")

    # Stats row
    if prev_step is not None:
        prev_correct = sum(1 for r in prev_results if r.get("is_correct"))
        prev_accuracy = prev_correct / len(prev_results) if prev_results else 0
        delta_accuracy = accuracy - prev_accuracy

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total", total)
        col2.metric("Correct", correct, delta=f"{correct - prev_correct:+d}")
        col3.metric("Accuracy", f"{accuracy:.1%}", delta=f"{delta_accuracy:+.1%}")
        col4.metric("🆕 Newly Solved", newly_solved)
        col5.metric("🔻 Newly Failed", newly_failed)

        st.caption(f"Compared to Step {prev_step}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Correct", correct)
        col3.metric("Accuracy", f"{accuracy:.1%}")
        st.caption("No previous step to compare")

    st.divider()

    # Result table
    if not results:
        st.info("No results match the current filter.")
        return

    # Initialize selected sample in session state
    if "selected_sample_idx" not in st.session_state:
        st.session_state.selected_sample_idx = None

    # Build table data
    st.subheader(f"Results ({len(results)} samples)")

    # Sort by index
    results = sorted(results, key=lambda r: r["index"])

    # Table with columns: Index, Status, Change, Prompt Brief, View
    for i, result in enumerate(results):
        idx = result["index"]
        is_correct = result.get("is_correct", False)
        prompt_brief = truncate_prompt(result.get("prompt", ""))

        # Determine change indicator
        change = step_diff.get(idx, "")
        if change == "new_pass":
            change_icon = "🆕"
            change_text = "Now solved"
        elif change == "new_fail":
            change_icon = "🔻"
            change_text = "Now failing"
        else:
            change_icon = ""
            change_text = ""

        status_icon = "✅" if is_correct else "❌"

        # Create row with columns
        cols = st.columns([0.5, 0.5, 1, 4, 1])

        cols[0].write(f"**#{idx}**")
        cols[1].write(status_icon)
        if change_icon:
            cols[2].write(f"{change_icon} {change_text}")
        else:
            cols[2].write("—")
        cols[3].write(prompt_brief)

        # View button
        if cols[4].button("View", key=f"view_{idx}"):
            st.session_state.selected_sample_idx = idx

    # Detail view
    st.divider()

    if st.session_state.selected_sample_idx is not None:
        # Find the result by index
        result = next(
            (r for r in all_results if r["index"] == st.session_state.selected_sample_idx),
            None,
        )

        if result:
            idx = result["index"]
            is_correct = result.get("is_correct", False)
            status = "✅ Correct" if is_correct else "❌ Incorrect"

            # Change info
            change = step_diff.get(idx, "")
            change_info = ""
            if change == "new_pass":
                change_info = " — 🆕 Newly Solved"
            elif change == "new_fail":
                change_info = " — 🔻 Newly Failed"

            st.subheader(f"Sample #{idx} — {status}{change_info}")

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

            # Clear button
            if st.button("Close Detail View"):
                st.session_state.selected_sample_idx = None
                st.rerun()
    else:
        st.info("👆 Click 'View' on a sample above to see details")


if __name__ == "__main__":
    main()
