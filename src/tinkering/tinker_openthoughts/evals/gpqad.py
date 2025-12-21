import asyncio
import random
import re
from pathlib import Path
import json
from typing import Any, Optional
from datasets import load_dataset
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
    ) -> dict[str, float]:
        sampling_params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.renderer.get_stop_sequences(),
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
                prompt=prompt, num_samples=1, sampling_params=sampling_params
            )
            return idx, res

        tasks = [wrapped_sample(i, p) for i, p in enumerate(prompts)]
        results = [None] * len(tasks)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
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

            response_tokens = r.sequences[0].tokens
            response_msg = self.renderer.parse_response(response_tokens)[0]
            content = renderers.ensure_text(response_msg["content"])
            correct_letter = correct_letters[i]
            extracted_answer = self._extract_answer(content)

            is_correct = extracted_answer == correct_letter
            if is_correct:
                num_correct += 1

            logged_results.append(
                {
                    "index": i,
                    "prompt": prompt_texts[i],
                    "response": content,
                    "correct_answer": correct_letter,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                }
            )

        if self.log_dir:
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            with open(log_path / "gpqa_diamond_results.jsonl", "w") as f:
                for res in logged_results:
                    f.write(json.dumps(res) + "\n")

        accuracy = num_correct / len(self.dataset) if self.dataset else 0
        return {"gpqa_diamond_accuracy": accuracy}


def gpqa_evaluator(renderer_name: str, model_name: str, log_dir: Optional[str] = None):
    """Builder function for the trainer."""
    return GPQADiamondEvaluator(
        model_name=model_name, renderer_name=renderer_name, log_dir=log_dir
    )
