import asyncio
import random
import re
from pathlib import Path
import json
from typing import Any, Optional
from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer


class GPQADiamondEvaluator(SamplingClientEvaluator):
    """
    GPQA Diamond Evaluator for multiple choice reasoning.
    """

    def __init__(
        self,
        model_name: str,
        renderer_name: str,
        max_samples: int = 100,  # GPQA Diamond has ~200 examples, we can limit for speed
        seed: int = 42,
        log_dir: Optional[str] = None,
        pass_at_k: int = 1,
    ):
        self.model_name = model_name
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

        # Load and subsample dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        if max_samples < len(ds):
            ds = ds.select(range(max_samples))
        self.dataset = ds
        self.seed = seed
        self.log_dir = log_dir
        self.pass_at_k = pass_at_k

    def _format_example(self, row: Any) -> tuple[str, str]:
        """Shuffles options and returns the prompt and the correct letter."""
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Use a fixed seed per question index for reproducibility across eval runs
        rng = random.Random(hash(row["Question"]) + self.seed)
        rng.shuffle(choices)

        letters = ["A", "B", "C", "D"]
        correct_letter = letters[choices.index(row["Correct Answer"])]

        options_str = "\n".join(
            [f"{letter}) {choice}" for letter, choice in zip(letters, choices)]
        )

        prompt = (
            f"Problem: {row['Question']}\n\n"
            f"Options:\n{options_str}\n\n"
            "Return your final response within \\boxed{} and only include the letter choice (A, B, C, or D) as your final response."
        )
        return prompt, correct_letter

    def _extract_answer(self, text: str) -> str:
        """Extracts the letter inside \boxed{}."""
        match = re.search(r"\\boxed{([A-D])}", text)
        if match:
            return match.group(1)
        # Fallback: check if the last character is a letter (common failure mode)
        return ""

    async def __call__(
        self,
        sampling_client: tinker.SamplingClient,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        step: Optional[int] = None,
    ) -> dict[str, float]:
        sampling_params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.renderer.get_stop_sequences(),
            seed=self.seed,
        )

        prompts = []
        prompt_texts = []
        correct_letters = []
        for row in self.dataset:
            prompt_text, correct_letter = self._format_example(row)
            model_input = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=prompt_text)]
            )
            prompts.append(model_input)
            prompt_texts.append(prompt_text)
            correct_letters.append(correct_letter)

        async def wrapped_sample(idx, prompt):
            res = await sampling_client.sample_async(
                prompt=prompt, num_samples=self.pass_at_k, sampling_params=sampling_params
            )
            return idx, res

        tasks = [wrapped_sample(i, p) for i, p in enumerate(prompts)]
        results = [None] * len(tasks)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=Console(force_terminal=True),
        ) as progress:
            eval_task = progress.add_task(
                "[cyan]Evaluating GPQA Diamond...", total=len(tasks)
            )
            # TODO: is this the most efficient way to set this up? I think tinker has it's own simpler abstraction for this
            for coro in asyncio.as_completed(tasks):
                idx, r = await coro
                results[idx] = r
                progress.advance(eval_task)

        num_correct = 0
        logged_results = []
        for i, r in enumerate(results):
            if r is None:
                continue

            correct_letter = correct_letters[i]
            
            # Check all k samples for pass@k
            sample_results = []
            any_correct = False
            for seq in r.sequences:
                response_tokens = seq.tokens
                response_msg = self.renderer.parse_response(response_tokens)[0]
                content = renderers.ensure_text(response_msg["content"])
                extracted_answer = self._extract_answer(content)
                is_sample_correct = extracted_answer == correct_letter
                sample_results.append({
                    "content": content,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_sample_correct,
                })
                if is_sample_correct:
                    any_correct = True

            if any_correct:
                num_correct += 1

            logged_results.append(
                {
                    "index": i,
                    "prompt": prompt_texts[i],
                    "samples": sample_results,
                    "correct_answer": correct_letter,
                    "pass_at_k": self.pass_at_k,
                    "is_correct": any_correct,
                }
            )

        if self.log_dir:
            log_path = Path(self.log_dir)
            if step is not None:
                log_path = log_path / f"step_{step:06d}"
            log_path.mkdir(parents=True, exist_ok=True)
            with open(log_path / "gpqa_diamond_results.jsonl", "w") as f:
                for res in logged_results:
                    res["step"] = step
                    f.write(json.dumps(res) + "\n")

        accuracy = num_correct / len(self.dataset) if self.dataset else 0
        metric_name = f"gpqa_diamond_pass@{self.pass_at_k}"
        return {metric_name: accuracy}


def gpqa_evaluator(
    renderer_name: str,
    model_name: str,
    log_dir: Optional[str] = None,
    pass_at_k: int = 1,
):
    """Builder function for the trainer."""
    return GPQADiamondEvaluator(
        model_name=model_name,
        renderer_name=renderer_name,
        log_dir=log_dir,
        pass_at_k=pass_at_k,
    )
