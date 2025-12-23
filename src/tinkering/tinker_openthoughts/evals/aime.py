"""
AIME 2025 Evaluator using tinker for sampling.

Based on evalchemy's AIME evaluation:
https://github.com/mlfoundations/evalchemy/blob/main/eval/chat_benchmarks/AIME25/eval_instruct.py
"""

import json
import re
from pathlib import Path
from typing import Optional

import tinker
from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Prompt template following evalchemy format
PROMPT_TEMPLATE = """Problem: {problem}
Mark your solution with \\boxed
Answer:"""


def extract_answer(output: str) -> str:
    """
    Extract the final answer from a model-generated solution.

    Tries multiple formats:
    1. \\boxed{answer} - LaTeX boxed format
    2. ANSWER: answer - Plain text format

    Returns empty string if no answer found.
    """
    # Try to extract from \boxed{...} - handle nested braces
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, output)
    if matches:
        return matches[-1].strip()

    # Try simple boxed without nested braces
    simple_boxed = r"\\boxed\{([^}]+)\}"
    matches = re.findall(simple_boxed, output)
    if matches:
        return matches[-1].strip()

    # Try ANSWER: format
    answer_pattern = r"ANSWER:\s*([^\n]+)"
    matches = re.findall(answer_pattern, output, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    # Remove common LaTeX formatting
    answer = answer.replace("$", "")
    answer = answer.replace(",", "")  # Remove comma separators
    answer = answer.strip()

    # Try to extract just the numeric part
    # AIME answers are always integers 0-999
    numeric_match = re.search(r"[-+]?\d+", answer)
    if numeric_match:
        return numeric_match.group()

    return answer


def is_correct(model_answer: str, target_answer: str) -> bool:
    """
    Check if the model answer matches the target answer.

    For AIME, answers are always integers 0-999.
    """
    model_normalized = normalize_answer(model_answer)
    target_normalized = normalize_answer(target_answer)

    # Direct string comparison after normalization
    if model_normalized == target_normalized:
        return True

    # Try numeric comparison
    try:
        return int(model_normalized) == int(target_normalized)
    except (ValueError, TypeError):
        return False


class AIME2025Evaluator(SamplingClientEvaluator):
    """
    AIME 2025 Evaluator for math reasoning.

    Based on evalchemy's evaluation approach, using tinker for sampling.
    """

    def __init__(
        self,
        model_name: str,
        renderer_name: str,
        max_samples: Optional[int] = None,
        log_dir: Optional[str] = None,
        seed: int = 42,
        pass_at_k: int = 1,
    ):
        self.model_name = model_name
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.log_dir = log_dir
        self.seed = seed
        self.pass_at_k = pass_at_k

        # Load AIME 2025 dataset
        ds = load_dataset("math-ai/aime25", split="test")

        self.dataset = []
        for row in ds:
            self.dataset.append(
                {
                    "id": row["id"],
                    "problem": row["problem"],
                    "answer": str(row["answer"]),
                }
            )

        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset[:max_samples]

    async def __call__(
        self,
        sampling_client: tinker.SamplingClient,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        step: Optional[int] = None,
    ) -> dict[str, float]:
        sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.renderer.get_stop_sequences(),
            seed=self.seed,
        )

        # Build prompts
        prompts = []
        prompt_texts = []
        for example in self.dataset:
            prompt_text = PROMPT_TEMPLATE.format(problem=example["problem"])
            model_input = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=prompt_text)]
            )
            prompts.append(model_input)
            prompt_texts.append(prompt_text)

        # Collect results
        num_correct = 0
        logged_results = []
        problem_results: dict[int, list[dict]] = {i: [] for i in range(len(prompts))}

        async def sample_one(idx: int, prompt):
            res = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=self.pass_at_k,
                sampling_params=sampling_params,
            )
            return idx, res

        import asyncio

        # Create all sampling tasks
        tasks = [asyncio.create_task(sample_one(i, p)) for i, p in enumerate(prompts)]

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=Console(force_terminal=True),
        ) as progress:
            task_id = progress.add_task(
                "[cyan]Evaluating AIME 2025...", total=len(prompts)
            )

            for coro in asyncio.as_completed(tasks):
                problem_idx, result = await coro
                progress.advance(task_id)

                example = self.dataset[problem_idx]
                target_answer = example["answer"]

                if result is None:
                    # No response - mark all samples as failed
                    for sample_idx in range(self.pass_at_k):
                        problem_results[problem_idx].append(
                            {
                                "sample_idx": sample_idx,
                                "content": "",
                                "model_answer": "",
                                "is_correct": False,
                                "error": "No response",
                            }
                        )
                else:
                    for sample_idx, seq in enumerate(result.sequences):
                        response_tokens = seq.tokens
                        response_msg = self.renderer.parse_response(response_tokens)[0]
                        content = renderers.ensure_text(response_msg["content"])

                        model_answer = extract_answer(content)
                        correct = is_correct(model_answer, target_answer)

                        problem_results[problem_idx].append(
                            {
                                "sample_idx": sample_idx,
                                "content": content,
                                "model_answer": model_answer,
                                "is_correct": correct,
                                "error": "",
                            }
                        )

        # Aggregate results per problem for pass@k
        for problem_idx in range(len(prompts)):
            samples = problem_results[problem_idx]
            any_correct = any(s["is_correct"] for s in samples)
            if any_correct:
                num_correct += 1

            logged_results.append(
                {
                    "index": problem_idx,
                    "problem_id": self.dataset[problem_idx]["id"],
                    "problem": self.dataset[problem_idx]["problem"],
                    "target_answer": self.dataset[problem_idx]["answer"],
                    "prompt": prompt_texts[problem_idx],
                    "samples": samples,
                    "pass_at_k": self.pass_at_k,
                    "is_correct": any_correct,
                }
            )

        # Save logs
        if self.log_dir:
            log_path = Path(self.log_dir)
            if step is not None:
                log_path = log_path / f"step_{step:06d}"
            log_path.mkdir(parents=True, exist_ok=True)
            with open(log_path / "aime2025_results.jsonl", "w") as f:
                for res in logged_results:
                    res["step"] = step
                    f.write(json.dumps(res) + "\n")

        accuracy = num_correct / len(self.dataset) if self.dataset else 0
        metric_name = f"aime2025_pass@{self.pass_at_k}"

        return {metric_name: accuracy}


def aime2025_evaluator(
    renderer_name: str,
    model_name: str,
    max_samples: int | None = None,
    log_dir: Optional[str] = None,
    seed: int = 42,
    pass_at_k: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 16384,
) -> SamplingClientEvaluator:
    """Builder function for the AIME 2025 evaluator."""
    return AIME2025Evaluator(
        model_name=model_name,
        renderer_name=renderer_name,
        max_samples=max_samples,
        log_dir=log_dir,
        seed=seed,
        pass_at_k=pass_at_k,
    )
