import jax.numpy as jnp

from jax.nn import log_softmax
from ml_collections.config_dict import ConfigDict


def get_config():
    config = ConfigDict({"figure_size": {"width": 10, "height": 5}, "metrics": {"rolling_average_window": 10}, "training": {"total_steps": 100000, "warmup_steps": 10000, "lr": 5e-5, "wd": 0.01, "b2": 0.95, "batch_size": 64}})
    return config


def cross_entropy_loss_and_accuracy(logits, tokens, valid):
    logits = logits.astype(jnp.float32)
    logp = log_softmax(logits)

    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)

    token_log_prob = jnp.squeeze(jnp.take_along_axis(logp, jnp.expand_dims(tokens, -1), -1), -1)
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))

    correct = jnp.where(valid > 0.0, jnp.argmax(logits, axis=-1) == tokens, jnp.array(False))
    # accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)

    loss = -(jnp.sum(token_log_prob) / jnp.sum(valid))
    metrics = {
        # "accuracy": accuracy,
        "token_logprob_sum": jnp.sum(token_log_prob),
        "valid_sum": jnp.sum(valid),
        "is_correct": jnp.sum(correct),
        "valid_text_length" : valid_text_length,
        
    }
    return loss, metrics
