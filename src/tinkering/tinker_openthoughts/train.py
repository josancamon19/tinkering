import logging
import asyncio
import time
from enum import Enum
import random

import chz
import numpy as np
import torch
from datasets import concatenate_datasets, load_from_disk, Dataset
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
import tinker
from pathlib import Path
from dotenv import load_dotenv
from tinker import types
from tinker_cookbook.utils.trace import scope, update_scope_context
from tinker_cookbook.supervised.train import (
    SubmittedBatch,
    compute_mean_nll,
)
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.renderers import get_renderer, TrainOnWhat

from tinkering.tinker_openthoughts.common import (
    openthoughts_row_to_datum,
    run_evals,
    compute_cosine_lr_with_warmup,
)
from tinkering.tinker_openthoughts.evals.aime import aime2025_evaluator
from tinkering.tinker_openthoughts.evals.gpqad import gpqa_evaluator
from tinkering.tinker_openthoughts.evals.livecodebench import livecodebench_evaluator
from tinkering.tinker_openthoughts.evals.nll import NLLEvaluator

load_dotenv()
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CurriculumMode(Enum):
    """Curriculum learning mode for ordering training data by difficulty."""

    NONE = "none"
    """No curriculum ordering (original behavior)"""

    EASY_TO_HARD = "easy_to_hard"
    """Order samples from lowest to highest difficulty"""

    FIRST_EPOCH_ONLY = "first_epoch_only"
    """Curriculum ordering on first epoch only, shuffle remaining epochs"""

    GROUPED_SHUFFLE = "grouped_shuffle"
    """Pool all epochs, group by difficulty, shuffle within each group, concat easy->hard"""


@chz.chz
class Config:
    wandb_project: str = "tinkering-openthoughts"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    # model_name: str = "Qwen/Qwen3-8B-Base"

    save_every: int = 20
    eval_every: int = 5
    infrequent_eval_every: int = 10  # cheap but slow, should run more often

    dataset_name: str = "openthoughts_code_all_sources_t4096_n100"
    train_split: float = 0.9
    pass_at_k: int = 1

    # hp's
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 5
    curriculum_mode: CurriculumMode = CurriculumMode.NONE
    lora_rank: int = 32
    # TODO: setup some hp tunning with some hp finding tool


def _setup_logging(config: Config, log_path: Path, run_name: str):
    return ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=run_name,
        config=config,
        do_configure_logging_module=True,
    )


def _validate_difficulty_field(dataset: Dataset) -> None:
    """Validate that all items in the dataset have a non-None difficulty value.

    Raises:
        ValueError: If any item has a None difficulty value.
    """
    if "difficulty" not in dataset.column_names:
        raise ValueError(
            "Curriculum mode requires a 'difficulty' column in the dataset, "
            "but no such column exists."
        )

    difficulties = dataset["difficulty"]
    none_indices = [i for i, d in enumerate(difficulties) if d is None]

    if none_indices:
        sample_indices = none_indices[:5]
        raise ValueError(
            f"Curriculum mode requires all items to have a non-None difficulty value. "
            f"Found {len(none_indices)} items with None difficulty. "
            f"Sample indices with None difficulty: {sample_indices}"
        )


def _prepare_curriculum_dataset(
    train_split: Dataset,
    curriculum_mode: CurriculumMode,
    epochs: int,
) -> Dataset:
    """Prepare the training dataset with curriculum ordering.

    Args:
        train_split: The training split of the dataset.
        curriculum_mode: The curriculum learning mode.
        epochs: Number of training epochs.

    Returns:
        The prepared training dataset with appropriate ordering.
    """
    if curriculum_mode == CurriculumMode.NONE:
        # Original behavior: just concatenate epochs
        return concatenate_datasets([train_split] * epochs)

    # Validate that all items have difficulty values
    _validate_difficulty_field(train_split)

    # Sort by difficulty (ascending = easy to hard)
    difficulties = train_split["difficulty"]
    sorted_indices = sorted(range(len(train_split)), key=lambda i: difficulties[i])
    ordered_train = train_split.select(sorted_indices)

    logger.info(
        f"Curriculum ordering applied: difficulty range {min(difficulties)} -> {max(difficulties)}"
    )

    if curriculum_mode == CurriculumMode.EASY_TO_HARD:
        # All epochs follow the same easy-to-hard order
        return concatenate_datasets([ordered_train] * epochs)

    elif curriculum_mode == CurriculumMode.FIRST_EPOCH_ONLY:
        # First epoch ordered, remaining epochs shuffled
        if epochs == 1:
            return ordered_train

        shuffled_epochs = concatenate_datasets(
            [train_split.shuffle(seed=42 + i) for i in range(1, epochs)]
        )
        return concatenate_datasets([ordered_train, shuffled_epochs])

    elif curriculum_mode == CurriculumMode.GROUPED_SHUFFLE:
        # Pool all epochs together, group by difficulty, shuffle within groups
        # This avoids consecutive repeats while maintaining easy->hard curriculum
        expanded_dataset = concatenate_datasets([train_split] * epochs)
        all_difficulties = expanded_dataset["difficulty"]

        # Group indices by difficulty level
        from collections import defaultdict
        import random

        difficulty_groups: dict[int, list[int]] = defaultdict(list)
        for idx, diff in enumerate(all_difficulties):
            difficulty_groups[diff].append(idx)

        # Shuffle within each difficulty group and concatenate in order
        random.seed(42)
        ordered_indices = []
        for difficulty in sorted(difficulty_groups.keys()):
            group_indices = difficulty_groups[difficulty]
            random.shuffle(group_indices)
            ordered_indices.extend(group_indices)

        logger.info(
            f"Grouped shuffle: {len(difficulty_groups)} difficulty levels, "
            f"{len(ordered_indices)} total samples"
        )

        return expanded_dataset.select(ordered_indices)

    else:
        raise ValueError(f"Unknown curriculum mode: {curriculum_mode}")


@scope
async def main(config: Config):
    # Set global seeds for reproducibility
    set_seed(42)

    if not Path(f"./subsets/{config.dataset_name}/dataset").exists():
        raise FileNotFoundError(f"Dataset {config.dataset_name} not found")

    # Construct the configuration name
    model_id = config.model_name.replace("/", "_")
    curriculum_suffix = (
        f"_c{config.curriculum_mode.value}"
        if config.curriculum_mode != CurriculumMode.NONE
        else ""
    )
    config_name = (
        f"{config.dataset_name}"
        f"_s{config.train_split}"
        f"_bs{config.batch_size}"
        f"_lr{config.learning_rate}"
        f"_e{config.epochs}"
        f"_r{config.lora_rank}"
        f"{curriculum_suffix}"
        f"_{model_id}"
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"./logs/openthoughts/{config_name}/{timestamp}")
    log_path.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(f"./subsets/{config.dataset_name}/dataset")
    split = dataset.train_test_split(train_size=config.train_split, seed=42)
    max_tokens = int(
        config.dataset_name.split("_t")[1].split("_")[0]
    )  # fixme: too hardcoded

    # Prepare train dataset with curriculum ordering (if enabled)
    train_dataset = _prepare_curriculum_dataset(
        train_split=split["train"],
        curriculum_mode=config.curriculum_mode,
        epochs=config.epochs,
    )
    test_dataset = split["test"]

    # Training loop counts
    batches_per_epoch = len(split["train"]) // config.batch_size
    total_steps = batches_per_epoch * config.epochs
    progress_denominator = max(total_steps, 1)

    ml_logger = _setup_logging(config, log_path, run_name=config_name)

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # TODO: couldn't figure the structure to pass as parameter
    evaluators = [
        NLLEvaluator.from_split(test_dataset, renderer, max_tokens, name="test")
    ]
    infrequent_evaluators = [
        # evaluator() for evaluator in config.infrequent_evaluator_builders
        aime2025_evaluator(
            renderer_name,
            config.model_name,
            log_dir=str(log_path / "inspect"),
            pass_at_k=1, # @k requires too many samples from inspect_utils.py assert num_responses == 1
        ),
        gpqa_evaluator(
            renderer_name,
            config.model_name,
            log_dir=str(log_path / "gpqa"),
            pass_at_k=config.pass_at_k,
        ),
        livecodebench_evaluator(
            renderer_name,
            config.model_name,
            max_samples=20,
            log_dir=str(log_path / "livecodebench"),
            pass_at_k=config.pass_at_k,
        ),
    ]

    @scope
    async def submit_batch(epoch_idx: int, batch_idx: int) -> SubmittedBatch:
        step = epoch_idx * batches_per_epoch + batch_idx
        update_scope_context({"step": step})

        batch_start_time = time.time()
        metrics: dict[str, int | float | str] = {"epoch": epoch_idx}
        metrics["progress"] = step / progress_denominator

        learning_rate = config.learning_rate * compute_cosine_lr_with_warmup(
            step=step, total_steps=total_steps, warmup_ratio=0.1
        )
        metrics["learning_rate"] = learning_rate

        adam_params = types.AdamParams(
            learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )

        with timed("get_batch", metrics):
            batch_start = step * config.batch_size
            batch_end = batch_start + config.batch_size
            rows = train_dataset.select(range(batch_start, batch_end))
            # Convert each row to a proper tinker.Datum with loss masking
            data: list[tinker.Datum] = [
                openthoughts_row_to_datum(
                    row,
                    renderer,
                    max_length=max_tokens,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                for row in rows
            ]

        # Trigger evaluations BEFORE submitting training operations so they snapshot pre-step weights
        eval_metrics = None
        if evaluators and config.eval_every > 0 and step % config.eval_every == 0:
            with timed("evals", metrics):
                eval_metrics = await run_evals(
                    evaluators, training_client, step, prefix="eval/"
                )

        infrequent_eval_metrics = None
        if (
            infrequent_evaluators
            and config.infrequent_eval_every > 0
            and step % config.infrequent_eval_every == 0
        ):
            with timed("infrequent_evals", metrics):
                infrequent_eval_metrics = await run_evals(
                    infrequent_evaluators, training_client, step, prefix="eval/"
                )

        fwd_bwd_future = await training_client.forward_backward_async(
            data, loss_fn="cross_entropy"
        )
        optim_step_future = await training_client.optim_step_async(adam_params)

        return SubmittedBatch(
            fwd_bwd_future=fwd_bwd_future,
            optim_step_future=optim_step_future,
            metrics=metrics,
            data=data,
            step=step,
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
            batch_start_time=batch_start_time,
            eval_metrics=eval_metrics,
            infrequent_eval_metrics=infrequent_eval_metrics,
        )

    @scope
    async def finish_batch(submitted: SubmittedBatch):
        update_scope_context({"step": submitted.step})

        metrics = submitted.metrics
        metrics["progress"] = min((submitted.step + 1) / progress_denominator, 1.0)

        if (
            config.save_every > 0
            and submitted.step % config.save_every == 0
            and submitted.step > 0
        ):
            with timed("save_checkpoint", metrics):
                # Enqueue a checkpoint save after the forward/backward and optimizer
                # requests for this step; the snapshot will reflect post-step weights.
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{submitted.step:06d}",
                    log_path=log_path,
                    loop_state={
                        "epoch": submitted.epoch_idx,
                        "batch": submitted.batch_idx,
                    },
                    kind="both",
                )

        with timed("step", metrics):
            fwd_bwd_result = await submitted.fwd_bwd_future.result_async()
            await submitted.optim_step_future.result_async()

        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in submitted.data]
        train_nll = compute_mean_nll(logprobs, weights)

        metrics.update(
            num_sequences=len(submitted.data),
            num_tokens=sum(datum.model_input.length for datum in submitted.data),
            num_loss_tokens=sum(
                sum(datum.loss_fn_inputs["weights"].data) for datum in submitted.data
            ),
            train_mean_nll=train_nll,
        )
        metrics["time/total"] = time.time() - submitted.batch_start_time

        # Merge evaluation metrics gathered before the training step was submitted
        if submitted.eval_metrics is not None:
            metrics.update(submitted.eval_metrics)

        if submitted.infrequent_eval_metrics is not None:
            metrics.update(submitted.infrequent_eval_metrics)

        # Emit all metrics for this step (train and eval) on the `submitted.step` row.
        ml_logger.log_metrics(metrics=metrics, step=submitted.step)

    pending_batch: SubmittedBatch | None = None
    start_batch, start_epoch = 0, 0

    for epoch_idx in range(start_epoch, config.epochs):
        logger.info(f"Starting epoch {epoch_idx}")

        start_batch_idx = start_batch if epoch_idx == start_epoch else 0
        for batch_idx in range(start_batch_idx, batches_per_epoch):
            submitted_batch = await submit_batch(epoch_idx, batch_idx)
            if pending_batch is not None:
                await finish_batch(pending_batch)
            pending_batch = submitted_batch

    if pending_batch is not None:
        await finish_batch(pending_batch)

    infrequent_evaluators[-1] = livecodebench_evaluator(
        renderer_name,
        config.model_name,
        max_samples=40,  # run a few more samples on final evaluation
        log_dir=str(log_path / "livecodebench"),
        pass_at_k=config.pass_at_k,
    )

    infrequent_eval_metrics = await run_evals(
        evaluators + infrequent_evaluators, training_client, total_steps, prefix="eval/"
    )
    ml_logger.log_metrics(infrequent_eval_metrics, step=total_steps)

    if start_epoch < config.epochs:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=log_path,
            kind="both",
            loop_state={"epoch": config.epochs, "batch": batches_per_epoch},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":

    def run(config: Config) -> None:
        asyncio.run(main(config))

    chz.nested_entrypoint(run, allow_hyphens=True)
