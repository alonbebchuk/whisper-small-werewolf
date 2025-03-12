import jax
import jax.numpy as jnp

from jax.nn import log_softmax

def loss_and_metrics(labels, logits, tokens, mask=None):
    logits = logits.astype(jnp.float32)
    
    if mask is None:
        mask = jnp.ones_like(tokens, dtype=jnp.float32)
    else:
        mask = mask.astype(jnp.float32)
        ones_indices = jnp.argmax((mask > 0).astype(jnp.int32), axis=-1)
        mask_shape = mask.shape
        mask = jnp.zeros_like(mask)
        mask = mask.at[jnp.arange(mask_shape[0]), ones_indices].set(1.0)

    total_sum = jnp.sum(mask)

    logp = log_softmax(logits)
    expanded_tokens = jnp.expand_dims(tokens, axis=-1)
    tokens_logp = jnp.take_along_axis(logp, expanded_tokens, axis=-1)
    tokens_logp = jnp.squeeze(tokens_logp, axis=-1)
    tokens_logp = jnp.where(mask > 0.0, tokens_logp, jnp.array(0.0))
    tokens_logp_sum = jnp.sum(tokens_logp)
    loss = -(tokens_logp_sum / total_sum)

    correct_logits = jnp.argmax(logits, axis=-1) == tokens
    correct_logits = jnp.where(mask > 0.0, correct_logits, jnp.array(False))
    correct_num = jnp.sum(correct_logits, axis=-1)
    correct = correct_num > 0
    
    tp = jnp.sum((correct & (labels == 1)))
    fp = jnp.sum((~correct & (labels == 0)))
    tn = jnp.sum((correct & (labels == 0)))
    fn = jnp.sum((~correct & (labels == 1)))
    
    n_pos = jnp.sum((labels == 1).astype(jnp.int32))
    n_neg = jnp.sum((labels == 0).astype(jnp.int32))

    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }

    return loss, metrics
