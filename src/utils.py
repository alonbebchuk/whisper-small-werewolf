import jax.numpy as jnp
def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32)  # for numerical stability
    logp = jax.nn.log_softmax(logits, axis=-1)
    
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            logp,
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -(jnp.sum(token_log_prob) / jnp.sum(valid))
    # old: loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    # changed to match hf implementation
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    metrics = {
        'accuracy': accuracy,
        'token_logprob_sum': jnp.sum(token_log_prob),
        'valid_sum': jnp.sum(valid),
    }
    return loss, metrics