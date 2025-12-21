import logging
import asyncio
import time
import chz
from datasets import concatenate_datasets, load_from_disk
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
import tinker
from pathlib import Path
from dotenv import load_dotenv
from tinker import types
from tinker_cookbook.utils.trace import scope, update_scope_context
import math
from tinker_cookbook.supervised.train import (
    EvaluatorBuilder,
    SubmittedBatch,
    compute_mean_nll,
    run_evals,
)
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.renderers import get_renderer, TrainOnWhat

from tinkering.tinker_openthoughts.common import openthoughts_row_to_datum
from tinkering.tinker_openthoughts.evals.nll import NLLEvaluator
from tinkering.tinker_openthoughts.evals.aime import aime2025_evaluator
from tinkering.tinker_openthoughts.evals.gpqad import gpqa_evaluator
from tinkering.tinker_openthoughts.evals.livecodebench import livecodebench_evaluator

load_dotenv()
logger = logging.getLogger(__name__)


@chz.chz
class Config:
    wandb_project: str = "tinkering-openthoughts"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    # model_name: str = "Qwen/Qwen3-8B-Base"

    infrequent_evaluator_builders: list[EvaluatorBuilder] = chz.field(
        default_factory=list
    )
    save_every: int = 20
    eval_every: int = 10
    infrequent_eval_every: int = 20

    dataset_name: str = "openthoughts_code_all_sources_t4096_n100"
    train_split: float = 0.8
    lora_rank: int = 32

    # hp's
    batch_size: int = 32
    learning_rate: float = 1e-5
    epochs: int = 13
    # TODO: setup some hp tunning with some hp finding tool


def compute_cosine_lr_with_warmup(
    step: int,
    total_steps: int,
    warmup_ratio: float = 0.1,
) -> float:
    """
    Compute learning rate multiplier with linear warmup and cosine decay.

    Args:
        step: Current training step (0-indexed)
        total_steps: Total number of training steps
        warmup_ratio: Fraction of total steps for warmup (default 10%)

    Returns:
        Learning rate multiplier in [0, 1]
    """
    warmup_steps = int(warmup_ratio * total_steps)

    if step < warmup_steps:
        # Linear warmup from 0 to 1
        return step / warmup_steps
    else:
        # Cosine decay from 1 to 0 over remaining steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))


def _setup_logging(config: Config, log_path: Path):
    return ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        config=config,
        do_configure_logging_module=True,
    )


@scope
async def main(config: Config):
    if not Path(f"./subsets/{config.dataset_name}/dataset").exists():
        raise FileNotFoundError(f"Dataset {config.dataset_name} not found")

    # Construct the configuration name
    model_id = config.model_name.replace("/", "_")
    config_name = (
        f"{config.dataset_name}"
        f"_s{config.train_split}"
        f"_bs{config.batch_size}"
        f"_lr{config.learning_rate}"
        f"_e{config.epochs}"
        f"_{model_id}"
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"./logs/openthoughts/{config_name}/{timestamp}")
    log_path.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(f"./subsets/{config.dataset_name}/dataset")
    split = dataset.train_test_split(train_size=config.train_split)
    max_tokens = int(config.dataset_name.split("sources_t")[1].split("_")[0])

    # Repeat train dataset for multiple epochs (simplifies batch indexing)
    train_dataset = concatenate_datasets([split["train"]] * config.epochs)
    test_dataset = split["test"]

    # Training loop counts
    batches_per_epoch = len(split["train"]) // config.batch_size
    total_steps = batches_per_epoch * config.epochs
    progress_denominator = max(total_steps, 1)

    ml_logger = _setup_logging(config, log_path)

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
        # NLLEvaluator.from_split(test_dataset, renderer, max_tokens, name="test")
    ]
    infrequent_evaluators = [
        # evaluator() for evaluator in config.infrequent_evaluator_builders
        # aime2025_evaluator(
        #     renderer_name,
        #     config.model_name,
        #     log_dir=str(log_path / "inspect"),
        # )
        # gpqa_evaluator(renderer_name, config.model_name, log_dir=str(log_path / "gpqa"))
        livecodebench_evaluator(renderer_name, config.model_name, max_samples=20, log_dir=str(log_path / "livecodebench"))
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
                eval_metrics = await run_evals(evaluators, training_client, step)

        infrequent_eval_metrics = None
        if (
            infrequent_evaluators
            and config.infrequent_eval_every > 0
            and step % config.infrequent_eval_every == 0
        ):
            with timed("infrequent_evals", metrics):
                infrequent_eval_metrics = await run_evals(
                    infrequent_evaluators, training_client, step
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
