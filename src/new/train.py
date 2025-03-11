# python3.10 -m src.new.train --model_name="bert"
# python3.10 -m src.new.train --model_name="whisper"
import os
import multiprocessing as mp

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
mp.set_start_method("spawn", force=True)

import datasets
import fire
import flax
import jax
import jax.numpy as jnp
import logging
import sys
import time
import transformers

from dataclasses import dataclass, field
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.metrics.tensorboard import SummaryWriter
from flax.training.common_utils import get_metrics, shard
from functools import partial
from pathlib import Path
from src.new.dataloaders import get_dataloaders
from src.new.models import get_model
from src.new.steps import get_steps
from src.new.train_state import get_train_state
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TrainingArguments

logger = logging.getLogger(__name__)


@flax.struct.dataclass
class ModelArguments:
    model_name: str


@dataclass
class UpdatedTrainingArgs(TrainingArguments):
    device_count: int = field(init=False)
    _train_batch_size: int = field(init=False)
    _eval_batch_size: int = field(init=False)
    num_examples: int = field(init=False)
    steps_per_epoch: int = field(init=False)
    num_train_steps: int = field(init=False)


def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def main():
    # 1. Parsing
    parser = HfArgumentParser((ModelArguments, UpdatedTrainingArgs))
    model_args, training_args = parser.parse_args_into_dataclasses()
    model_name = model_args.model_name

    # 2. Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # 3. Training Arguments I
    training_args.device_count = jax.device_count()
    training_args._train_batch_size = training_args.per_device_train_batch_size * training_args.device_count
    training_args._eval_batch_size = training_args.per_device_eval_batch_size * training_args.device_count

    # 4. Train and Eval Loaders
    train_loader, eval_loader = get_dataloaders(model_name, training_args)

    # 5. Training Arguments II
    training_args.num_examples = len(train_loader.dataset)
    training_args.steps_per_epoch = training_args.num_examples // training_args._train_batch_size
    training_args.num_train_steps = training_args.steps_per_epoch * training_args.num_train_epochs

    # 6. Log Training Arguments
    logger.info("Training parameters %s", training_args)

    # 7. Train State
    state = get_train_state(model_name, training_args)
    state = state.replicate()

    # 8. Train and Eval Steps
    train_step, eval_step = get_steps(model_name)
    p_train_step = jax.pmap(partial(train_step), "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(partial(eval_step), "batch")

    # 9. Training
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {training_args.num_examples}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {training_args._train_batch_size}")
    logger.info(f"  Total optimization steps = {training_args.num_train_steps}")

    train_time = 0
    epochs = tqdm(range(int(training_args.num_train_epochs)), desc=f"Epoch ... (1/{training_args.num_train_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []
        for batch in tqdm(train_loader, desc="Training...", position=1, leave=False):
            batch = shard(batch)
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)
        train_time += time.time() - train_start
        train_metric = unreplicate(train_metric)
        epochs.write(f"Epoch... ({epoch + 1}/{training_args.num_train_epochs} | Loss: {train_metric['loss']}, Learning Rate:" f" {train_metric['learning_rate']})")

        # ======================== Evaluating ==============================
        eval_metrics = []
        for batch in tqdm(eval_loader, desc="Evaluating...", position=2, leave=False):
            metrics = pad_shard_unpad(p_eval_step, static_return=True)(state, batch, min_device_batch=training_args.per_device_eval_batch_size)
            eval_metrics.append(metrics)
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)
        desc = f"Epoch... ({epoch + 1}/{training_args.num_train_epochs} | Eval Loss: {eval_metrics['loss']} | {eval_metrics})"
        epochs.write(desc)
        epochs.desc = desc

        # Save metrics
        if jax.process_index() == 0:
            cur_step = epoch * training_args.steps_per_epoch
            write_metric(summary_writer, train_metrics, eval_metrics, train_time, cur_step)

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            model = get_model(model_name)
            model.save_pretrained(training_args.output_dir, params=params)


if __name__ == "__main__":
    fire.Fire(main)
