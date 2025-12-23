#!/usr/bin/env python3
"""
Script to compare GPQA results across different timestamps for the same settings.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dicts."""
    results = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def get_timestamps(base_dir: Path) -> List[str]:
    """Get all timestamp directories in the base directory."""
    return sorted([
        d.name for d in base_dir.iterdir()
        if d.is_dir() and d.name.isdigit() or d.name.replace('_', '').isdigit()
    ])


def find_gpqa_results(timestamp_dir: Path, step: int) -> Path | None:
    """Find the GPQA results file for a given timestamp and step."""
    step_str = f"step_{step:06d}"
    gpqa_path = timestamp_dir / "gpqa" / step_str / "gpqa_diamond_results.jsonl"
    if gpqa_path.exists():
        return gpqa_path
    return None


def extract_question_key(result: Dict[str, Any]) -> str:
    """Extract a unique key for the question (ignoring option ordering)."""
    # Use the problem part of the prompt (before options)
    prompt = result.get('prompt', '')
    # Extract just the problem statement (before Options:)
    if 'Options:' in prompt:
        problem = prompt.split('Options:')[0].strip()
    else:
        problem = prompt
    return problem


def compare_results(
    base_dir: Path,
    step: int = 0
) -> Dict[str, Any]:
    """
    Compare GPQA results across all timestamps for a given step.
    
    Returns a dict with:
    - summary: {timestamp: {passed, failed, total}}
    - differences: list of questions where results differ
    """
    timestamps = get_timestamps(base_dir)
    
    if len(timestamps) < 2:
        print(f"Need at least 2 timestamps to compare, found: {timestamps}")
        return {}
    
    # Load results from all timestamps
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for ts in timestamps:
        ts_dir = base_dir / ts
        gpqa_path = find_gpqa_results(ts_dir, step)
        if gpqa_path:
            all_results[ts] = load_jsonl(gpqa_path)
            print(f"Loaded {len(all_results[ts])} results from {ts}")
        else:
            print(f"Warning: No GPQA results found for {ts} at step {step}")
    
    if len(all_results) < 2:
        print("Not enough results to compare")
        return {}
    
    # Compute summary for each timestamp
    summary = {}
    for ts, results in all_results.items():
        passed = sum(1 for r in results if r.get('is_correct', False))
        total = len(results)
        summary[ts] = {
            'passed': passed,
            'failed': total - passed,
            'total': total,
            'accuracy': passed / total if total > 0 else 0
        }
    
    # Find differences by index
    differences = []  # Different outcomes (pass vs fail)
    same_outcome_diff_response = []  # Same outcome but different responses
    
    # Get all indices from first timestamp as reference
    first_ts = list(all_results.keys())[0]
    indices = {r['index']: r for r in all_results[first_ts]}
    
    for idx in sorted(indices.keys()):
        results_by_ts = {}
        for ts, results in all_results.items():
            # Find result with matching index
            for r in results:
                if r['index'] == idx:
                    results_by_ts[ts] = r
                    break
        
        if len(results_by_ts) < 2:
            continue
        
        # Extract response content for each timestamp
        responses_content = {}
        for ts, r in results_by_ts.items():
            samples = r.get('samples', [])
            if samples:
                responses_content[ts] = samples[0].get('content', '')
            else:
                responses_content[ts] = ''
        
        # Check if is_correct differs between any timestamps
        is_correct_values = [r.get('is_correct', False) for r in results_by_ts.values()]
        
        # Build the diff object
        diff = {
            'index': idx,
            'prompt': results_by_ts[first_ts].get('prompt', ''),
            'correct_answer': {},
            'extracted_answer': {},
            'is_correct': {},
            'responses': {}
        }
        
        for ts, r in results_by_ts.items():
            diff['correct_answer'][ts] = r.get('correct_answer', '')
            samples = r.get('samples', [])
            if samples:
                diff['extracted_answer'][ts] = samples[0].get('extracted_answer', '')
                diff['responses'][ts] = samples[0].get('content', '')
            else:
                diff['extracted_answer'][ts] = ''
                diff['responses'][ts] = ''
            diff['is_correct'][ts] = r.get('is_correct', False)
        
        if len(set(is_correct_values)) > 1:
            # Different outcomes (pass vs fail)
            differences.append(diff)
        else:
            # Same outcome - check if responses differ
            response_texts = list(responses_content.values())
            if len(set(response_texts)) > 1:
                # Same outcome but different responses
                outcome = "PASSED" if is_correct_values[0] else "FAILED"
                diff['outcome'] = outcome
                same_outcome_diff_response.append(diff)
    
    return {
        'summary': summary,
        'differences': differences,
        'same_outcome_diff_response': same_outcome_diff_response,
        'timestamps': timestamps
    }


def save_comparison(
    comparison: Dict[str, Any],
    output_dir: Path,
    step: int = 0
) -> None:
    """Save comparison results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_path = output_dir / f"step_{step:06d}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"GPQA Results Comparison - Step {step}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Summary by Timestamp:\n")
        f.write("-" * 40 + "\n")
        for ts, stats in comparison['summary'].items():
            f.write(f"\n{ts}:\n")
            f.write(f"  Passed: {stats['passed']}/{stats['total']} ({stats['accuracy']:.1%})\n")
            f.write(f"  Failed: {stats['failed']}/{stats['total']}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"\nTotal outcome differences found: {len(comparison['differences'])}\n")
        
        same_diff = comparison.get('same_outcome_diff_response', [])
        both_passed = [d for d in same_diff if d.get('outcome') == 'PASSED']
        both_failed = [d for d in same_diff if d.get('outcome') == 'FAILED']
        f.write(f"Same outcome, different response: {len(same_diff)} total\n")
        f.write(f"  - Both passed, different response: {len(both_passed)}\n")
        f.write(f"  - Both failed, different response: {len(both_failed)}\n")
        
        if comparison['differences']:
            f.write("\nQuestions with different outcomes:\n")
            f.write("-" * 40 + "\n")
            for diff in comparison['differences']:
                f.write(f"\nIndex {diff['index']}:\n")
                for ts in comparison['timestamps']:
                    if ts in diff['is_correct']:
                        status = "✓ PASS" if diff['is_correct'][ts] else "✗ FAIL"
                        ans = diff['extracted_answer'].get(ts, 'N/A')
                        correct = diff['correct_answer'].get(ts, 'N/A')
                        f.write(f"  {ts}: {status} (answered: {ans}, correct: {correct})\n")
    
    print(f"Saved summary to {summary_path}")
    
    # Save each difference as separate files for easy tab comparison
    if comparison['differences']:
        diffs_dir = output_dir / f"step_{step:06d}_differences"
        diffs_dir.mkdir(parents=True, exist_ok=True)
        
        for diff in comparison['differences']:
            q_dir = diffs_dir / f"question_{diff['index']:03d}"
            q_dir.mkdir(parents=True, exist_ok=True)
            
            # Save input/prompt as separate file
            input_path = q_dir / "0_input.txt"
            with open(input_path, 'w') as f:
                f.write(f"Question Index: {diff['index']}\n")
                f.write("=" * 80 + "\n\n")
                f.write("PROMPT:\n")
                f.write("-" * 80 + "\n")
                f.write(diff['prompt'] + "\n")
            
            # Save each timestamp's response as separate file
            for i, ts in enumerate(comparison['timestamps'], 1):
                if ts not in diff['is_correct']:
                    continue
                
                status = "PASSED" if diff['is_correct'][ts] else "FAILED"
                response_path = q_dir / f"{i}_{ts}_{status}.txt"
                
                with open(response_path, 'w') as f:
                    f.write(f"Question Index: {diff['index']}\n")
                    f.write(f"Timestamp: {ts}\n")
                    f.write(f"Status: {'✓ PASSED' if diff['is_correct'][ts] else '✗ FAILED'}\n")
                    f.write(f"Extracted Answer: {diff['extracted_answer'].get(ts, 'N/A')}\n")
                    f.write(f"Correct Answer: {diff['correct_answer'].get(ts, 'N/A')}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("RESPONSE:\n")
                    f.write("-" * 80 + "\n")
                    f.write(diff['responses'].get(ts, 'N/A') + "\n")
        
        print(f"Saved {len(comparison['differences'])} difference folders to {diffs_dir}")
    
    # Save same-outcome but different-response cases
    same_diff = comparison.get('same_outcome_diff_response', [])
    if same_diff:
        same_dir = output_dir / f"step_{step:06d}_same_outcome_diff_response"
        same_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate by outcome type
        both_passed = [d for d in same_diff if d.get('outcome') == 'PASSED']
        both_failed = [d for d in same_diff if d.get('outcome') == 'FAILED']
        
        for outcome, items in [('both_passed', both_passed), ('both_failed', both_failed)]:
            if not items:
                continue
            
            outcome_dir = same_dir / outcome
            outcome_dir.mkdir(parents=True, exist_ok=True)
            
            for diff in items:
                q_dir = outcome_dir / f"question_{diff['index']:03d}"
                q_dir.mkdir(parents=True, exist_ok=True)
                
                # Save input/prompt as separate file
                input_path = q_dir / "0_input.txt"
                with open(input_path, 'w') as f:
                    f.write(f"Question Index: {diff['index']}\n")
                    f.write(f"Outcome: {diff.get('outcome', 'N/A')} (both timestamps)\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("PROMPT:\n")
                    f.write("-" * 80 + "\n")
                    f.write(diff['prompt'] + "\n")
                
                # Save each timestamp's response as separate file
                for i, ts in enumerate(comparison['timestamps'], 1):
                    if ts not in diff['is_correct']:
                        continue
                    
                    response_path = q_dir / f"{i}_{ts}.txt"
                    
                    with open(response_path, 'w') as f:
                        f.write(f"Question Index: {diff['index']}\n")
                        f.write(f"Timestamp: {ts}\n")
                        f.write(f"Status: {'✓ PASSED' if diff['is_correct'][ts] else '✗ FAILED'}\n")
                        f.write(f"Extracted Answer: {diff['extracted_answer'].get(ts, 'N/A')}\n")
                        f.write(f"Correct Answer: {diff['correct_answer'].get(ts, 'N/A')}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write("RESPONSE:\n")
                        f.write("-" * 80 + "\n")
                        f.write(diff['responses'].get(ts, 'N/A') + "\n")
        
        print(f"Saved {len(both_passed)} both-passed and {len(both_failed)} both-failed diff-response folders to {same_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare GPQA results across timestamps"
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing timestamp subdirectories"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Training step to compare (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for comparison files (default: base_dir/comparisons)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "comparisons"
    
    print(f"Comparing GPQA results in: {base_dir}")
    print(f"Step: {args.step}")
    print(f"Output: {output_dir}")
    print()
    
    comparison = compare_results(base_dir, args.step)
    
    if comparison:
        save_comparison(comparison, output_dir, args.step)
        print("\nDone!")
    else:
        print("No comparison results generated")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

