import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import jax
import numpy as np
import wandb


from src.common.config import get_config, get_model_name
from src.common.lr_schedule import create_learning_rate_schedule
from src.data.data_stream import DataStream
from src.evaluation.eval_step import eval_step
from src.models.whisper import FlaxWhisperForConditionalGeneration
from src.training.train_state import create_train_state
from src.training.train_step import train_step
from tqdm.auto import tqdm


def train(use_audio, use_dialogue):
    config = get_config()

    model_name = get_model_name(config, use_audio, use_dialogue)

    worker_id = jax.process_index()
    if worker_id == 0:
        wandb.init(entity="alonbebchuk-tel-aviv-university", project=model_name, config=config.to_dict())

    stream = DataStream(config, use_audio, use_dialogue)

    lr_schedule = create_learning_rate_schedule(config)

    model = FlaxWhisperForConditionalGeneration.from_pretrained(config.model.name, from_pt=True)

    state = create_train_state(config, model, lr_schedule)
    state = state.replicate()

    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=tuple())

    pbar = tqdm(range(config.training.total_steps), desc="Training")
    eval_counter = config.evaluation.eval_freq
    for step, batch in zip(pbar, stream.train_iter()):
        print(jax.tree.map(np.shape, batch))
        epoch = batch.pop("epoch", 0)

        state, curr_loss, curr_acc = p_train_step(state, batch)
        curr_loss = curr_loss.mean().item()
        curr_acc = curr_acc.mean().item()

        pbar.set_description(f"Loss: {curr_loss:.4f}, Acc: {curr_acc:.4f}")
        metrics = {"step": step, "loss": float(curr_loss), "accuracy": float(curr_acc), "lr": float(lr_schedule(step)), "epoch": epoch}
        print(metrics)

        eval_counter -= 1
        if eval_counter == 0:
            eval_counter = config.evaluation.eval_freq
            for i, dev_batch in enumerate(stream.validation_iter()):
                if i >= config.evaluation.eval_steps:
                    break
                dev_batch.pop("epoch", 0)
                curr_loss, curr_acc = p_eval_step(state, dev_batch)
                curr_loss = curr_loss.mean().item()
                curr_acc = curr_acc.mean().item()

            if worker_id == 0:
                wandb.log({"eval_loss": curr_loss, "eval_accuracy": curr_acc, "epoch": epoch, "step": step})
            print({"eval_loss": curr_loss, "eval_accuracy": curr_acc, "epoch": epoch, "step": step})

        if worker_id == 0:
            wandb.log(metrics)
    if worker_id == 0:
        wandb.finish()
        params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
        model.save_pretrained(os.path.join(os.path.dirname(__file__), f"model/{model_name}"), params=params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_audio", type=bool, required=True)
    parser.add_argument("--use_dialogue", type=bool, required=True)
    args = parser.parse_args()

    train(args.use_audio, args.use_dialogue)
