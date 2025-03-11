import jax.numpy as jnp


from jax import jit, value_and_grad
from jax.lax import psum
from src.common.loss_and_metrics import loss_and_metrics
from src.training.train_state import TrainStateWithMetrics
from typing import Dict

import jax.numpy as jnp
from jax import jit
import optax
from flax.training.common_utils import onehot



@jit
def train_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        outputs = state.apply_fn(**{"params": params},
                                 input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 train=True,
                                 dropout_rng=state.dropout_rng)

        # preds = jnp.argmax(outputs.logits, axis=-1)
        loss, metrics = loss_and_metrics(outputs.logits, batch["labels"])
        preds = None
        return loss, (preds, metrics)

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, (preds, metrics)), grads = grad_fn(state.params)


    # del preds
    grads = psum(grads, "batch")
    new_state = state.apply_gradients(grads=grads)

    loss = psum(loss, "batch")
    curr_loss, new_loss = new_state.loss_metric.update(loss)

    metrics = psum(metrics, "batch")
    acc = metrics["correct_sum"] / metrics["total_sum"]
    curr_acc, new_acc = new_state.acc_metric.update(acc)

    new_state = new_state.replace(loss_metric=new_loss, acc_metric=new_acc)

    return new_state, curr_loss, curr_acc, preds




@jit
def eval_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        outputs = state.apply_fn(**{"params": params}, 
                                input_ids=batch["input_ids"], 
                                attention_mask=batch["attention_mask"], 
                                deterministic=True)

        loss, metrics = loss_and_metrics(outputs.logits, batch["labels"], batch["attention_mask"])
        return loss, metrics

    loss, metrics = loss_fn(state.params)

    loss = psum(loss, "batch")
    metrics = psum(metrics, "batch")
    acc = metrics["correct_sum"] / metrics["total_sum"]

    return loss, acc
