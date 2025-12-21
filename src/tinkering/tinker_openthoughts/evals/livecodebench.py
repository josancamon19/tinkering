from typing import Optional


import tinker
from tinker_cookbook.eval.evaluators import (
    SamplingClientEvaluator,
)

from tinkering.tinker_openthoughts.evals.lcb_utils import (
    lcb_run,
    map_to_example,
    post_process_code,
    translate_private_test_cases,
)
import re
import json
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from pathlib import Path


class LiveCodeBenchEvaluator(SamplingClientEvaluator):
    """
    LiveCodeBench Evaluator for code generation.
    """

    def __init__(
        self,
        model_name: str,
        renderer_name: str,
        max_samples: Optional[int] = None,
        log_dir: Optional[str] = None,
        seed: int = 42,
    ):
        self.model_name = model_name
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.log_dir = log_dir
        self.seed = seed

        # Load dataset
        # Note: Using version_tag="release_v5" as in reference
        lcb_codegen = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag="release_v5",
            split="test",
        )

        def filter_by_contest_date(example):
            target_months = [
                "2024-08",
                "2024-09",
                "2024-10",
                "2024-11",
                "2024-12",
                "2025-01",
            ]
            return example["contest_date"][:7] in target_months

        ds = lcb_codegen.filter(filter_by_contest_date)

        # Process examples
        processed_examples = []
        for row in ds:
            row_copy = copy.deepcopy(row)
            row_copy["private_test_cases"] = translate_private_test_cases(
                row_copy["private_test_cases"]
            )
            processed_examples.append(map_to_example(row_copy))

        if max_samples and max_samples < len(processed_examples):
            processed_examples = processed_examples[:max_samples]

        self.dataset = processed_examples

    def _extract_code(self, response: str) -> Optional[str]:
        pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1]  # Use the last code block
        return None

    async def __call__(
        self,
        sampling_client: tinker.SamplingClient,
        temperature: float = 0.0,
        max_tokens: int = 16384,
    ) -> dict[str, float]:
        sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.renderer.get_stop_sequences(),
        )

        prompts = []
        prompt_texts = []
        for example in self.dataset:
            if example["is_stdin"]:
                prompt_text = (
                    "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition.\n"
                    + example["prompt"]
                )
            else:
                prompt_text = (
                    "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution.\n"
                    + example["prompt"]
                )

            model_input = self.renderer.build_generation_prompt(
                [renderers.Message(role="user", content=prompt_text)]
            )
            prompts.append(model_input)
            prompt_texts.append(prompt_text)

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
                "[cyan]Sampling LiveCodeBench...", total=len(tasks)
            )
            for coro in asyncio.as_completed(tasks):
                idx, r = await coro
                results[idx] = r
                progress.advance(eval_task)

        # Evaluation phase
        def evaluate_one(idx):
            r = results[idx]
            example = self.dataset[idx]
            if r is None:
                return idx, False, "No response", ""

            response_tokens = r.sequences[0].tokens
            response_msg = self.renderer.parse_response(response_tokens)[0]
            content = renderers.ensure_text(response_msg["content"])

            code = self._extract_code(content)
            if not code:
                return idx, False, "No code found", content

            # Run execution validation
            try:
                # lcb_run is blocking, but we are in a thread pool
                result_list = lcb_run(
                    problem=example,
                    completion=post_process_code(code),
                    timeout=6,
                    is_extracted=not example["is_stdin"],
                )
                is_correct = all(r[0] for r in result_list)
                return idx, is_correct, "", content
            except Exception as e:
                return idx, False, str(e), content

        num_correct = 0
        logged_results = []

        with ThreadPoolExecutor(max_workers=32) as executor:
            loop = asyncio.get_running_loop()
            eval_futures = [
                loop.run_in_executor(executor, evaluate_one, i)
                for i in range(len(results))
            ]

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ) as progress:
                exec_task = progress.add_task(
                    "[cyan]Executing LiveCodeBench...", total=len(eval_futures)
                )
                for fut in asyncio.as_completed(eval_futures):
                    idx, is_correct, error, content = await fut
                    if is_correct:
                        num_correct += 1

                    logged_results.append(
                        {
                            "index": idx,
                            "prompt": prompt_texts[idx],
                            "response": content,
                            "is_correct": is_correct,
                            "error": error,
                            "difficulty": self.dataset[idx]["difficulty"],
                        }
                    )
                    progress.advance(exec_task)

        if self.log_dir:
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            with open(log_path / "livecodebench_results.jsonl", "w") as f:
                for res in logged_results:
                    f.write(json.dumps(res) + "\n")

        accuracy = num_correct / len(self.dataset) if self.dataset else 0
        return {"livecodebench_accuracy": accuracy}


def livecodebench_evaluator(
    renderer_name: str, model_name: str, log_dir: Optional[str] = None
):
    """Builder function for the trainer."""
    return LiveCodeBenchEvaluator(
        model_name=model_name, renderer_name=renderer_name, log_dir=log_dir
    )


# TODO: once it runs, how to easily visualize traces, and trajectories
