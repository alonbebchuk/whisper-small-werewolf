import os

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_cache"


import datasets
import jax
import logging
import numpy as np
import sys
import time
import transformers

from datasets import DatasetDict, load_dataset
from flax.training.common_utils import shard
from src.data.data_collaters import BertDataCollator, WhisperDataCollator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import send_example_telemetry


logger = logging.getLogger(__name__)


def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one token to the right.
    """
    shifted_label_ids = np.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id

    return shifted_label_ids


def main(do_train, do_eval, data_collator):
    send_example_telemetry("run_speech_recognition_seq2seq", framework="flax")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    raw_datasets = DatasetDict()

    if do_train:
        raw_datasets["train"] = load_dataset("alonbeb/werewolf-data", split="train", num_proc=10)

    if do_eval:
        raw_datasets["eval"] = load_dataset("alonbeb/werewolf-data", split="validation", num_proc=10)

    num_epochs = 2
    per_device_train_batch_size = 8
    train_batch_size = per_device_train_batch_size * jax.device_count()
    per_device_eval_batch_size = 8
    eval_batch_size = per_device_eval_batch_size * jax.device_count()

    epochs = tqdm(range(num_epochs), desc=f"Epoch (1/{num_epochs})", position=0)
    train_total_time = 0
    eval_total_time = 0
    for epoch in epochs:
        # ======================== Training ================================
        train_epoch_start = time.time()

        train_loader = DataLoader(raw_datasets["train"], batch_size=train_batch_size, shuffle=True, num_workers=10, collate_fn=data_collator, drop_last=True)
        # train
        for batch in tqdm(train_loader, desc="Training...", position=1, leave=False):
            batch = shard(batch)
            print(jax.tree.map(np.shape, batch))

        train_epoch_time = time.time() - train_epoch_start
        train_total_time += train_epoch_time

        epochs.write(f"Train Epoch: {epoch + 1} | Train Epoch Time: {train_epoch_time}s")

        # ======================== Evaluating ==============================
        eval_epoch_start = time.time()

        eval_loader = DataLoader(raw_datasets["eval"], batch_size=eval_batch_size, shuffle=True, num_workers=10, collate_fn=data_collator, drop_last=True)
        for batch in tqdm(eval_loader, desc="Evaluating...", position=2, leave=False):
            batch = shard(batch)
            print(jax.tree.map(np.shape, batch))

        eval_epoch_time = time.time() - eval_epoch_start
        eval_total_time += eval_epoch_time

        epochs.write(f"Eval Epoch: {epoch + 1} | Eval Epoch Time: {eval_epoch_time}s")

        epochs.desc = f"Epoch ({epoch + 1}/{num_epochs})"

    epochs.write(f"Total Time: {train_total_time + eval_total_time} | Train Total Time: {train_total_time} | Eval Total Time: {eval_total_time}")


if __name__ == "__main__":
    print("Bert Test:")
    main(True, True, BertDataCollator())

    print("Whisper Test:")
    main(True, True, WhisperDataCollator())
