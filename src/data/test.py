import os

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_cache"

import datasets
import jax
import logging
import multiprocessing as mp
import numpy as np
import sys
import time
import transformers

from datasets import load_from_disk
from flax.training.common_utils import shard
from src.data.data_collators import BertDataCollator, WhisperDataCollator
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    level = logging.INFO if jax.process_index() == 0 else logging.ERROR
    logger.setLevel(level)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


def worker_init_fn(worker_id):
    np.random.seed(worker_id + int(time.time()))


def create_dataloaders(dataset, data_collator):
    device_count = jax.device_count()
    train_batch_size = 8 * device_count
    eval_batch_size = 8 * device_count
    num_workers = os.cpu_count() // 2
    prefetch_factor = 4

    common_kwargs = dict(
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    train_loader = DataLoader(dataset["train"], batch_size=train_batch_size, shuffle=True, **common_kwargs)
    validation_loader = DataLoader(dataset["validation"], batch_size=eval_batch_size, shuffle=False, **common_kwargs)
    return {"train": train_loader, "validation": validation_loader}


def run_epoch(loader, desc, position):
    start_time = time.time()
    first_batch_logged = False
    for batch in tqdm(loader, desc=desc, position=position, leave=False):
        batch = shard(batch)
        if not first_batch_logged:
            shapes = jax.tree.map(lambda x: x.shape, batch)
            print("First batch shapes:", shapes)
            first_batch_logged = True
    return time.time() - start_time


def main(data_collator):
    setup_logging()

    dataset = load_from_disk("/dev/shm/hf_cache/werewolf_data")
    dataloaders = create_dataloaders(dataset, data_collator)

    train_num_rows = dataset["train"].num_rows
    validation_num_rows = dataset["validation"].num_rows

    num_epochs = 2
    total_train_time, total_eval_time = 0, 0

    epochs = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epochs:
        train_time = run_epoch(dataloaders["train"], desc=f"Training Epoch {epoch + 1}", position=1)
        eval_time = run_epoch(dataloaders["validation"], desc=f"Evaluating Epoch {epoch + 1}", position=2)
        total_train_time += train_time
        total_eval_time += eval_time
        logger.info(f"Epoch {epoch + 1}: Training {train_num_rows / train_time:.2f} examples/s, Evaluating {validation_num_rows / eval_time:.2f} examples/s")

    logger.info(f"Total: Training {num_epochs * train_num_rows / total_train_time:.2f} examples/s, Evaluating {num_epochs * validation_num_rows / total_eval_time:.2f} examples/s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    logger.info("Running Bert Test...")
    main(BertDataCollator())

    logger.info("Running Whisper Test...")
    main(WhisperDataCollator())
