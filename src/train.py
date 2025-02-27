import jax
import numpy as np


from src.common.utils import get_config
from src.common.whisper import FlaxWhisperForConditionalGeneration
from src.data.data import create_stream
from tqdm.auto import tqdm
from flax.training.common_utils import shard
from src.training.train_state import create_train_state
from src.training.train_step import train_step


config = get_config()
model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-small", from_pt=True)
state, lr_schedule = create_train_state(config, model)
state = state.replicate()
p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
# p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=tuple())
total_steps = config.training.total_steps
pbar = tqdm(range(total_steps), desc="Training")
eval_freq = 2000
eval_steps = 5
eval_counter = eval_freq
seen_examples = 0
stream = create_stream()
import wandb
wandb.init(project="whisper-small-werewolf")
for step, batch in zip(pbar, stream):
    print(jax.tree.map(np.shape, batch))
    batch = shard(batch)
    epoch = batch.pop("epoch", 0)

    state, curr_loss, curr_acc = p_train_step(state, batch)
    # total_n_examples = int(total_n_examples[0])
    # seen_examples += total_n_examples
    curr_loss = curr_loss.mean().item()
    curr_acc = curr_acc.mean().item()

    pbar.set_description(f"Loss: {curr_loss:.4f}, Acc: {curr_acc:.4f}")
    metrics = {
        "step": step,
        "loss": float(curr_loss),
        "accuracy": float(curr_acc),
        "lr": float(lr_schedule(step)),
        "epoch": epoch,
        # "seen_examples": seen_examples,
    }
    wandb.log(metrics)

    # print(metrics)

    # eval_counter -= 1
    # if eval_counter==0:
    #     eval_counter = eval_freq
    #     for i, dev_batch in enumerate(stream.validation_iter()):
    #         if i>=eval_steps:
    #             break
    #         dev_batch.pop("epoch", 0)
    #         curr_loss, curr_acc = p_eval_step(state, dev_batch)
    #         curr_loss = curr_loss.mean().item()
    #         curr_acc = curr_acc.mean().item()