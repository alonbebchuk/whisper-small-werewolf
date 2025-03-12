import jax.numpy as jnp
import jax

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

        preds = jnp.argmax(outputs.logits, axis=-1)
        loss, metrics = loss_and_metrics(batch["labels"], outputs.logits, batch["target_tokens"], batch["loss_mask"])
        return loss, (metrics, preds)

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, preds)), grads = grad_fn(state.params)

    grads = psum(grads, "batch")
    new_state = state.apply_gradients(grads=grads)

    loss = psum(loss, "batch")
    curr_loss, new_loss = new_state.loss_metric.update(loss)

    metrics = psum(metrics, "batch")
    tp = metrics["tp"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    tn = metrics["tn"]
    n_pos = metrics["n_pos"]
    n_neg = metrics["n_neg"]
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    tpr = recall
    fpr = fp / (fp + tn + 1e-10)
    roc_auc = (tpr * n_pos + (1 - fpr) * n_neg) / (n_pos + n_neg + 1e-10)
    
    jax.debug.print("precision: {x}", x=precision)
    jax.debug.print("recall: {x}", x=recall)
    jax.debug.print("f1: {x}", x=f1)
    jax.debug.print("accuracy: {x}", x=accuracy)
    jax.debug.print("tpr: {x}", x=tpr)
    jax.debug.print("fpr: {x}", x=fpr)
    jax.debug.print("roc_auc: {x}", x=roc_auc)
    

    curr_acc, new_acc = new_state.acc_metric.update(accuracy)

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

        loss, metrics = loss_and_metrics(batch["labels"], outputs.logits, batch["target_tokens"], batch["loss_mask"])
        return loss, metrics

    loss, metrics = loss_fn(state.params)

    loss = psum(loss, "batch")
    metrics = psum(metrics, "batch")

    return loss, metrics
