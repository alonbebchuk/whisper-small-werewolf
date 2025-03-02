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
