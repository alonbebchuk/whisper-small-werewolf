import jax
import jax.numpy as jnp
import numpy as np

from src.common.utils import cross_entropy_loss_and_accuracy
from jax import jit, value_and_grad
from jax.lax import psum, pmean
from src.training.train_state import TrainStateWithMetrics
from typing import Dict


@jit
def train_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        outputs = state.apply_fn(**{"params": params}, input_features=batch["input_features"], decoder_input_ids=batch["decoder_input_ids"], decoder_attention_mask=batch["attention_mask"], train=True)

        logits = outputs.logits
        unnorm_loss, metrics = cross_entropy_loss_and_accuracy(logits, batch["target_tokens"], batch["loss_mask"])
        return unnorm_loss, metrics

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (unnorm_loss, metrics), grads = grad_fn(state.params)

    grads = psum(grads, "batch")
    metrics = psum(metrics,"batch")
    new_state = state.apply_gradients(grads=grads)

    # total_n_examples = psum(logits.shape[0], "batch")
    # total_is_correct = psum(metrics["is_correct"], "batch")
    total_loss = psum(unnorm_loss, "batch")

    # acc = total_is_correct / total_n_examples
    # loss = total_loss / total_n_examples

    # acc = pmean(metrinp["accuracy"], "batch")
    # print(jax.tree.map(jnp.shape, metrics))
    acc = metrics["is_correct"] / metrics["valid_sum"]

    curr_loss, new_loss_metric = new_state.loss_metric.update(total_loss)
    curr_acc, new_acc_metric = new_state.acc_metric.update(acc)

    new_state = new_state.replace(loss_metric=new_loss_metric, acc_metric=new_acc_metric)

    return new_state, curr_loss, curr_acc
