from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from flax.training.train_state import TrainState
from jax.random import PRNGKey, split
import optax
from src.common.rolling_avg import RollingAverage


class TrainStateWithMetrics(TrainState):
    loss_metric: RollingAverage
    acc_metric: RollingAverage
    dropout_rng: PRNGKey

    def replicate(self):
        replicated = replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))
        return replicated


from flax import struct, traverse_util

# We use Optax's "masking" functionality to not apply weight decay
# to bias and LayerNorm scale parameters. decay_mask_fn returns a
# mask boolean with the same structure as the parameters.
# The mask is True for parameters that should be decayed.
def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
    layer_norm_named_params = {
        layer[-2:]
        for layer_norm_name in layer_norm_candidates
        for layer in flat_params.keys()
        if layer_norm_name in "".join(layer).lower()
    }
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)

def create_train_state(config, model, lr_schedule):
    apply_fn = model.__call__
    params = model.params
    opt = optax.adamw(lr_schedule, weight_decay=config.training.wd, b2=config.training.b2, eps=float(config.training.eps), mask=decay_mask_fn)
    opt = optax.chain(optax.clip_by_global_norm(1), opt)
    tx = optax.apply_if_finite(opt, max_consecutive_errors=1000000)
    # tx = optax.chain(opt, optax.skip_not_finite)
    loss_metric = RollingAverage.create(size=config.metrics.rolling_average_window)
    acc_metric = RollingAverage.create(size=config.metrics.rolling_average_window)
    dropout_rng = split(PRNGKey(0))[1]

    train_state = TrainStateWithMetrics.create(apply_fn=apply_fn, params=params, tx=tx, loss_metric=loss_metric, acc_metric=acc_metric, dropout_rng=dropout_rng)
    return train_state
