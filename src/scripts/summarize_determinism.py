#!/usr/bin/env python3
"""
Summarize determinism analysis across all tested models.
"""

import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelResult:
    model_name: str
    model_type: str  # Full, Instruct, Base
    accuracy_range: str
    outcome_flips: int
    identical_pct: float
    total_questions: int


def parse_summary(summary_path: Path) -> dict:
    """Parse a summary file and extract key metrics."""
    content = summary_path.read_text()
    
    # Extract accuracies
    accuracy_matches = re.findall(r'Passed: (\d+)/(\d+) \((\d+\.\d+)%\)', content)
    accuracies = [float(m[2]) for m in accuracy_matches]
    total = int(accuracy_matches[0][1]) if accuracy_matches else 100
    
    # Extract outcome differences
    outcome_match = re.search(r'Total outcome differences found: (\d+)', content)
    outcome_flips = int(outcome_match.group(1)) if outcome_match else 0
    
    # Extract same outcome different response
    same_diff_match = re.search(r'Same outcome, different response: (\d+) total', content)
    same_diff = int(same_diff_match.group(1)) if same_diff_match else 0
    
    # Calculate identical responses
    different_responses = outcome_flips + same_diff
    identical = total - different_responses
    identical_pct = (identical / total * 100) if total > 0 else 0
    
    return {
        'accuracies': accuracies,
        'outcome_flips': outcome_flips,
        'same_diff': same_diff,
        'identical': identical,
        'identical_pct': identical_pct,
        'total': total
    }


def get_model_type(model_name: str) -> str:
    """Determine model type from name."""
    if '-Instruct' in model_name:
        return 'Instruct'
    elif '-Base' in model_name:
        return 'Base'
    else:
        return 'Full'


def extract_model_name(path: Path) -> str:
    """Extract clean model name from path."""
    # Path like: .../openthoughts_all_domains_..._meta-llama_Llama-3.1-8B/comparisons/...
    parts = path.parts
    for part in parts:
        if part.startswith('openthoughts_'):
            # Extract last part after the last underscore pattern for model
            # e.g., "meta-llama_Llama-3.1-8B" or "Qwen_Qwen3-8B"
            match = re.search(r'_([^_]+_[^_]+)$', part)
            if match:
                return match.group(1).replace('_', '/')
    return "Unknown"


def main():
    logs_dir = Path("logs/openthoughts")
    summaries = list(logs_dir.glob("*/comparisons/step_000000_summary.txt"))
    
    results = []
    for summary_path in sorted(summaries):
        model_name = extract_model_name(summary_path)
        model_type = get_model_type(model_name)
        metrics = parse_summary(summary_path)
        
        if len(metrics['accuracies']) >= 2:
            acc_min = min(metrics['accuracies'])
            acc_max = max(metrics['accuracies'])
            if acc_min == acc_max:
                accuracy_range = f"{acc_min:.0f}%"
            else:
                accuracy_range = f"{acc_min:.0f}-{acc_max:.0f}%"
        else:
            accuracy_range = "N/A"
        
        results.append(ModelResult(
            model_name=model_name,
            model_type=model_type,
            accuracy_range=accuracy_range,
            outcome_flips=metrics['outcome_flips'],
            identical_pct=metrics['identical_pct'],
            total_questions=metrics['total']
        ))
    
    # Sort by identical_pct descending
    results.sort(key=lambda x: x.identical_pct, reverse=True)
    
    # Print table
    print("\n" + "=" * 90)
    print("DETERMINISM ANALYSIS SUMMARY - GPQA Diamond (Step 0)")
    print("=" * 90)
    print(f"\n{'Model':<35} {'Type':<10} {'Accuracy':<12} {'Flips':<8} {'Identical':<12}")
    print("-" * 90)
    
    for r in results:
        print(f"{r.model_name:<35} {r.model_type:<10} {r.accuracy_range:<12} {r.outcome_flips:<8} {r.identical_pct:.0f}%")
    
    print("-" * 90)
    print("\nNotes:")
    print("- Flips: Questions where pass/fail outcome differed between runs")
    print("- Identical: Percentage of questions with exactly identical responses")
    print("- All runs used temperature=0 and seed=42")
    print()
    
    # Create ASCII bar chart
    print("\n" + "=" * 90)
    print("DETERMINISM CHART (% Identical Responses)")
    print("=" * 90 + "\n")
    
    max_bar = 50
    for r in results:
        bar_len = int(r.identical_pct / 100 * max_bar)
        bar = "█" * bar_len + "░" * (max_bar - bar_len)
        print(f"{r.model_name:<30} |{bar}| {r.identical_pct:.0f}%")
    
    print()
    
    # Save to file
    output_path = Path("logs/openthoughts/determinism_summary.txt")
    with open(output_path, 'w') as f:
        f.write("DETERMINISM ANALYSIS SUMMARY - GPQA Diamond (Step 0)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Model':<35} {'Type':<10} {'Accuracy':<12} {'Flips':<8} {'Identical':<12}\n")
        f.write("-" * 90 + "\n")
        for r in results:
            f.write(f"{r.model_name:<35} {r.model_type:<10} {r.accuracy_range:<12} {r.outcome_flips:<8} {r.identical_pct:.0f}%\n")
        f.write("-" * 90 + "\n")
        f.write("\nDETERMINISM CHART (% Identical Responses)\n")
        f.write("=" * 90 + "\n\n")
        for r in results:
            bar_len = int(r.identical_pct / 100 * max_bar)
            bar = "█" * bar_len + "░" * (max_bar - bar_len)
            f.write(f"{r.model_name:<30} |{bar}| {r.identical_pct:.0f}%\n")
    
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()

