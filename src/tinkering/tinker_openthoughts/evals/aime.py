from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_evaluators import (
    InspectEvaluator,
    InspectEvaluatorBuilder,
)


def aime2025_evaluator(
    renderer_name: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    limit: int | None = None,
    log_dir: str | None = None,
) -> SamplingClientEvaluator:
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
