import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import jax
import jax.numpy as jnp
import numpy as np
import wandb


from src.common.config import get_config
from src.common.lr_schedule import create_learning_rate_schedule
from transformers import FlaxBertForSequenceClassification, FlaxWhisperForConditionalGeneration
from src.training.train_state import create_train_state
import src.training.whisper_steps as whisper_steps
import src.training.bert_steps as bert_steps
from tqdm.auto import tqdm


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


def get_data_collator(model_name):
    if "bert" in model_name:
        model = FlaxBertForSequenceClassification.from_pretrained(model_name, from_pt=True)
        return BertDataCollator(), bert_steps, model
    else:
        assert "whisper" in model_name
        model = FlaxWhisperForConditionalGeneration.from_pretrained(model_name, from_pt=True)
        return WhisperDataCollator(), whisper_steps, model


def train(model_name):
    setup_logging()
    mp.set_start_method("spawn", force=True)
    config = get_config(model_name)
    dataset = load_from_disk("/dev/shm/hf_cache/werewolf_data")

    device_count = jax.device_count()
    train_batch_size = 8 * device_count
    num_workers = os.cpu_count() // 2
    prefetch_factor = 4
    data_collator, steps, model = get_data_collator(model_name)

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
    num_epochs = 2
    worker_id = jax.process_index()
    if worker_id == 0:
        wandb.init(project="whisper_werewolf", config=config.to_dict())

    lr_schedule = create_learning_rate_schedule(config)

    state = create_train_state(config, model, lr_schedule)
    state = state.replicate()

    p_train_step = jax.pmap(steps.train_step, "batch", donate_argnums=(0,))

    pbar = tqdm(range(config.training.total_steps), desc="Training")
    epochs = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epochs:
        for step, batch in zip(pbar, train_loader):
            print(jax.tree.map(np.shape, batch))
            epoch = batch.pop("epoch", 0)
            strategy = batch.pop("strategy")
            batch = shard(batch)

            state, curr_loss, curr_acc, preds = p_train_step(state, batch)
            if step > 100:
                print(f"{preds=}")
                assert preds is not None
            curr_loss = curr_loss.mean().item()
            curr_acc = curr_acc.mean().item()

            pbar.set_description(f"Loss: {curr_loss:.4f}, Acc: {curr_acc:.4f}")
            metrics = {"step": step, "loss": float(curr_loss), "accuracy": float(curr_acc), "lr": float(lr_schedule(step)), "epoch": epoch}
            print(metrics)

            if worker_id == 0:
                preds = jax.device_get(preds)
                if "bert" in model_name:
                    targets = jax.device_get(batch["labels"])
                    mask = jnp.ones_like(targets)
                else:
                    assert "whisper" in model_name
                    targets = jax.device_get(batch["target_tokens"])
                    mask = jax.device_get(batch["loss_mask"])                
                strategy = jax.device_get(strategy)

                print(f"Predictions shape: {preds.shape}")
                print(f"Targets shape: {targets.shape}")
                print(f"Mask shape: {mask.shape}")
                print(f"Strategy shape: {strategy.shape}")

                preds_2d = preds.reshape(-1, preds.shape[-1])
                targets_2d = targets.reshape(-1, targets.shape[-1])
                mask_2d = mask.reshape(-1, mask.shape[-1])

                print(f"Reshaped predictions shape: {preds_2d.shape}")
                print(f"Reshaped targets shape: {targets_2d.shape}")
                print(f"Reshaped mask shape: {mask_2d.shape}")

                strategy_metrics = {}
                unique_strategies = np.unique(strategy)
                print(f"Unique strategies found: {unique_strategies}")

                for strat in unique_strategies:
                    print(f"\nProcessing strategy: {strat}")
                    strat_indices = (strategy == strat)
                    
                    strat_preds = preds_2d[strat_indices]
                    strat_targets = targets_2d[strat_indices]
                    strat_mask = mask_2d[strat_indices]
                    
                    print(f"Strategy {strat} shapes:")
                    print(f"- Predictions: {strat_preds.shape}")
                    print(f"- Targets: {strat_targets.shape}")
                    print(f"- Mask: {strat_mask.shape}")
                    
                    correct_logits = (strat_preds == strat_targets)
                    correct_logits = np.where(strat_mask > 0.0, correct_logits, np.array(False))
                    correct_sum = np.sum(correct_logits)
                    total_sum = np.sum(strat_mask)
                    
                    if total_sum > 0:
                        accuracy = float(correct_sum) / float(total_sum)
                        print(f"Strategy {strat} metrics:")
                        print(f"- Correct: {correct_sum}")
                        print(f"- Total: {total_sum}")
                        print(f"- Accuracy: {accuracy:.4f}")
                        strategy_metrics[f"accuracy_{strat}"] = accuracy
                        strategy_metrics[f"correct_{strat}"] = int(correct_sum)
                        strategy_metrics[f"total_{strat}"] = int(total_sum)

                metrics.update(strategy_metrics)
                wandb.log(metrics)

    if worker_id == 0:
        wandb.finish()
        # params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
        # model.save_pretrained(os.path.join(os.path.dirname(__file__), f"model/{model_name}"), params=params)


import fire

# python3.10 -m src.train --model_name=openai/whisper-small
# python3.10 -m src.train --model_name=google-bert/bert-base-cased
if __name__ == "__main__":
    fire.Fire(train)
