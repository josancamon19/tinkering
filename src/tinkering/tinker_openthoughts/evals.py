from typing import Optional

from datasets import Dataset

import tinker
from tinker_cookbook.eval.evaluators import (
    TrainingClientEvaluator,
    SamplingClientEvaluator,
)
from tinker_cookbook.eval.inspect_evaluators import (
    InspectEvaluator,
    InspectEvaluatorBuilder,
)
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.renderers import Renderer, TrainOnWhat

from tinkering.tinker_openthoughts.common import openthoughts_row_to_datum


class NLLEvaluator(TrainingClientEvaluator):
    def __init__(self, data: list[tinker.Datum], name: str = "test"):
        self.name = name
        self.data = data

    async def __call__(
        self, training_client: tinker.TrainingClient
    ) -> dict[str, float]:
        future = await training_client.forward_async(self.data, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]
        nll = compute_mean_nll(logprobs, weights)
        key = f"{self.name}/nll"
        return {key: nll}

    @classmethod
    def from_split(
        cls,
        split: Dataset,
        renderer: Renderer,
        max_tokens: int,
        name: str = "test",
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> "NLLEvaluator":
        """Create an NLLEvaluator from a dataset split (e.g., split["test"])."""
        data = [
            openthoughts_row_to_datum(
                row, renderer, max_length=max_tokens, train_on_what=train_on_what
            )
            for row in split
        ]
        return cls(data, name=name)


def aime2025_evaluator(
    renderer_name: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    limit: Optional[int] = None,
    log_dir: Optional[str] = None,
) -> SamplingClientEvaluator:
    """
    Create an evaluator for AIME 2025 (American Invitational Mathematics Examination).

    This is a simple wrapper that uses inspect_evals/aime2025 task via InspectEvaluator.

    Args:
        renderer_name: Name of the renderer to use (e.g., "llama3", "qwen3")
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        temperature: Sampling temperature (default 0.0 for greedy decoding)
        max_tokens: Maximum tokens to generate (default 32768 for long math reasoning)
        limit: Optional limit on number of samples to evaluate
        log_dir: Optional directory for inspect logs

    Returns:
        A SamplingClientEvaluator that can be used in training configs
    """
    config = InspectEvaluatorBuilder(
        tasks="inspect_evals/aime2025",
        renderer_name=renderer_name,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        limit=limit,
        log_dir=log_dir,
    )
    return InspectEvaluator(config)
