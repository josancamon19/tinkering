from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat, Message
from tinker_cookbook.supervised.common import datum_from_model_input_weights
import tinker
import math
import logging
import asyncio
from dotenv import load_dotenv
from tinker_cookbook.utils.trace import scope, update_scope_context

from tinker_cookbook.eval.evaluators import (
    Evaluator,
    SamplingClientEvaluator,
    TrainingClientEvaluator,
)


def openthoughts_row_to_datum(
    row: dict,
    renderer: renderers.Renderer,
    max_length: int | None = None,
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> tinker.Datum:
    """
    Convert an OpenThoughts dataset row to a tinker.Datum.

    OpenThoughts format:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "<think>...</think>..."}
        ]
    }

    This converts to chat Message format and uses the renderer to build
    a supervised example with proper loss masking (only train on assistant response).
    """
    conversations = row["conversations"]
    messages: list[Message] = []

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append(Message(role=role, content=turn["value"]))

    # Build the supervised example using the renderer
    # This handles tokenization, masking, and proper formatting for the model
    # weights it's such a bad name for a mask.!!!
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=train_on_what
    )

    return datum_from_model_input_weights(model_input, weights, max_length)


load_dotenv()
logger = logging.getLogger(__name__)


@scope
async def run_evals(
    evaluators: list[Evaluator],
    training_client: tinker.TrainingClient,
    step: int,
    prefix: str = "",
) -> dict[str, float]:
    """
    Run evaluators in parallel and pass step to each one.

    This is a custom version that passes `step` to evaluators so they can
    include it in their logs for tracing training evolution.

    Args:
        evaluators: List of evaluators to run
        training_client: The training client
        step: Current training step
    """
    update_scope_context({"step": step})

    # Check if any evaluators need a sampling client
    sampling_evaluators = [
        e for e in evaluators if isinstance(e, SamplingClientEvaluator)
    ]

    # Create sampling client upfront if any sampling evaluators exist
    sampling_client = None
    if sampling_evaluators:
        sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async(
                f"evals_step_{step}"
            )
        )

    @scope
    async def run_evaluator(evaluator: Evaluator) -> dict[str, float]:
        update_scope_context(
            {
                "step": step,
                "evaluator_name": type(evaluator).__name__,
            }
        )
        if isinstance(evaluator, TrainingClientEvaluator):
            update_scope_context({"evaluator_type": "TrainingClientEvaluator"})
            try:
                return await evaluator(training_client, step=step)
            except TypeError:
                return await evaluator(training_client)
        elif isinstance(evaluator, SamplingClientEvaluator):
            update_scope_context({"evaluator_type": "SamplingClientEvaluator"})
            try:
                return await evaluator(sampling_client, step=step)
            except TypeError:
                return await evaluator(sampling_client)
        else:
            raise ValueError(f"Unknown evaluator type: {type(evaluator)}")

    # Run all evaluators in parallel
    all_results = await asyncio.gather(*[run_evaluator(e) for e in evaluators])

    # Merge all metrics with prefix
    metrics = {}
    accuracy_values = []
    for eval_metrics in all_results:
        for key, value in eval_metrics.items():
            prefixed_key = f"{prefix}{key}" if prefix else key
            metrics[prefixed_key] = value
            # Collect accuracy metrics for averaging
            if "accuracy" in key.lower() and isinstance(value, (int, float)):
                accuracy_values.append(value)

    # Add average accuracy if we have multiple accuracy metrics
    if prefix and accuracy_values:
        metrics[f"{prefix}avg_accuracy"] = sum(accuracy_values) / len(accuracy_values)

    return metrics


def compute_cosine_lr_with_warmup(
    step: int, total_steps: int, warmup_ratio: float = 0.1
) -> float:
    warmup_steps = int(warmup_ratio * total_steps)

    if step < warmup_steps:
        # Linear warmup from 0 to 1
        return step / warmup_steps
    else:
        # Cosine decay from 1 to 0 over remaining steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
