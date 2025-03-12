import jax.numpy as jnp

from jax.nn import log_softmax
import jax
from optax import sigmoid_binary_cross_entropy
def loss_and_metrics(logits, tokens, mask=None):
    logits = logits.astype(jnp.float32)
    
    if mask is None:
        # Create a mask of ones with the same shape as tokens
        mask = jnp.ones_like(tokens, dtype=jnp.float32)
    else:
        mask = mask.astype(jnp.float32)

    total_sum = jnp.sum(mask)
    
    
    logp = log_softmax(logits)
    expanded_tokens = jnp.expand_dims(tokens, axis=-1)
    tokens_logp = jnp.take_along_axis(logp, expanded_tokens, axis=-1)
    # jax.debug.print("🤯 tokens={tokens}", tokens=tokens)
    
    tokens_logp = jnp.where(jnp.isnan(tokens_logp), 0, tokens_logp)
    # has_nan = jnp.any(jnp.isnan(logits), axis=-1)
    tokens_logp = jnp.squeeze(tokens_logp, axis=-1)
    tokens_logp = jnp.where(mask > 0.0, tokens_logp, jnp.array(0.0))
    tokens_logp_sum = jnp.sum(tokens_logp)
    loss = -(tokens_logp_sum / total_sum)

    correct_logits = jnp.argmax(logits, axis=-1) == tokens
    correct_logits = jnp.where(mask > 0.0, correct_logits, jnp.array(False))
    correct_sum = jnp.sum(correct_logits)
    metrics = {"correct_sum": correct_sum, "total_sum": total_sum}
    jax.debug.print("🤯 loss={loss}", loss=loss)

    return loss, metrics
