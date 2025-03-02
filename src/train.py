import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import jax

# import wandb


from src.common.config import get_config, get_strategy_dataset_name
from src.common.lr_schedule import create_learning_rate_schedule
from src.data.data_stream import DataStream
from src.evaluation.eval_step import eval_step
from src.model.whisper import FlaxWhisperForConditionalGeneration
from src.training.train_state import create_train_state
from src.training.train_step import train_step
from tqdm.auto import tqdm


def main(strategy):
    config = get_config()

    # worker_id = jax.process_index()
    # if worker_id == 0:
    #     wandb.init(project=f"vit_jax_{get_strategy_dataset_name(config, strategy)}", config=config.to_dict())

    stream = DataStream(config, strategy)

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

            # if worker_id == 0:
            #     wandb.log({"eval_loss": curr_loss, "eval_accuracy": curr_acc, "epoch": epoch, "step": step})
            print({"eval_loss": curr_loss, "eval_accuracy": curr_acc, "epoch": epoch, "step": step})

    #     if worker_id == 0:
    #         wandb.log(metrics)
    # if worker_id == 0:
    #     wandb.finish()


if __name__ == "__main__":
    main("Accusation")
