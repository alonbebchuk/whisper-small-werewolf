import jax
import jax.numpy as jnp
import optax

from flax import jax_utils, struct, traverse_util
from flax.training import train_state
from flax.training.common_utils import onehot, shard_prng_key
from src.new.learning_rate import get_learning_rate_fn
from src.new.models import get_model
from typing import Callable


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray
    learning_rate_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)
    logits_fn: Callable = struct.field(pytree_node=False)

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def get_adamw(training_args, learning_rate_fn):
    return optax.adamw(
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        learning_rate=learning_rate_fn,
    )


def get_dropout_rng(training_args):
    rng = jax.random.PRNGKey(training_args.seed)
    _, dropout_rng = jax.random.split(rng)
    return dropout_rng


def create_train_state(model_name, training_args, loss_fn):
    model = get_model(model_name)
    learning_rate_fn = get_learning_rate_fn(training_args)
    adamw = get_adamw(training_args, learning_rate_fn)
    dropout_rng = get_dropout_rng(training_args)

    return TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=adamw,
        dropout_rng=dropout_rng,
        learning_rate_fn=learning_rate_fn,
        loss_fn=loss_fn,
        logits_fn=lambda logits: logits.argmax(-1),
    )


def get_bert_train_state(model_name, training_args):
    num_classes = 2

    def loss_fn(logits, labels):
        xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes))
        return jnp.mean(xentropy)

    return create_train_state(model_name, training_args, loss_fn)


def get_whisper_train_state(model_name, training_args):
    def loss_fn(logits, labels):
        vocab_size = logits.shape[-1]
        confidence = 1.0 - training_args.label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)
        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant
        padding_mask = labels >= 0
        loss = loss * padding_mask
        loss = loss.sum()
        num_labels = padding_mask.sum()
        return loss, num_labels

    return create_train_state(model_name, training_args, loss_fn)


_train_state = None


def get_train_state(model_name, training_args):
    global _train_state
    if _train_state is None:
        if model_name == "bert":
            _train_state = get_bert_train_state(model_name, training_args)
        elif model_name == "whisper":
            _train_state = get_whisper_train_state(model_name, training_args)
        else:
            raise Exception(f"Model name {model_name} is not supported.")
    return _train_state
