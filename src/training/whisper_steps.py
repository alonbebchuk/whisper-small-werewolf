import jax.numpy as jnp


from jax import jit, value_and_grad
from jax.lax import psum
from src.common.loss_and_metrics import loss_and_metrics
from src.training.train_state import TrainStateWithMetrics
from typing import Dict




@jit
def train_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        outputs = state.apply_fn(**{"params": params},
                                 input_features=batch["input_features"],
                                 decoder_input_ids=batch["decoder_input_ids"],
                                 decoder_attention_mask=batch["attention_mask"], train=True)

        loss, metrics = loss_and_metrics(outputs.logits, batch["target_tokens"], batch["loss_mask"])
        preds = None
        return loss, (metrics, preds)

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, preds)), grads = grad_fn(state.params)

    grads = psum(grads, "batch")
    new_state = state.apply_gradients(grads=grads)

    loss = psum(loss, "batch")
    curr_loss, new_loss = new_state.loss_metric.update(loss)

    metrics = psum(metrics, "batch")
    acc = metrics["correct_sum"] / metrics["total_sum"]
    curr_acc, new_acc = new_state.acc_metric.update(acc)

    new_state = new_state.replace(loss_metric=new_loss, acc_metric=new_acc)

    return new_state, curr_loss, curr_acc, preds


import jax.numpy as jnp

from jax import jit
from jax.lax import psum
from src.common.loss_and_metrics import loss_and_metrics
from src.training.train_state import TrainStateWithMetrics
from typing import Dict


@jit
def eval_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        outputs = state.apply_fn(**{"params": params}, input_features=batch["input_features"], decoder_input_ids=batch["decoder_input_ids"], decoder_attention_mask=batch["attention_mask"], train=True)

        loss, metrics = loss_and_metrics(outputs.logits, batch["target_tokens"], batch["loss_mask"])
        return loss, metrics

    loss, metrics = loss_fn(state.params)

    loss = psum(loss, "batch")
    metrics = psum(metrics, "batch")
    acc = metrics["correct_sum"] / metrics["total_sum"]

    return loss, acc
