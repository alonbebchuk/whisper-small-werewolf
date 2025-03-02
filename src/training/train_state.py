from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from flax.training.train_state import TrainState
from jax.random import PRNGKey, split
from optax import adamw
from src.common.rolling_avg import RollingAverage


class TrainStateWithMetrics(TrainState):
    loss_metric: RollingAverage
    acc_metric: RollingAverage
    dropout_rng: PRNGKey

    def replicate(self):
        replicated = replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))
        return replicated


def create_train_state(config, model, lr_schedule):
    apply_fn = model.__call__
    params = model.params
    tx = adamw(lr_schedule, weight_decay=config.training.wd, b2=config.training.b2)
    loss_metric = RollingAverage.create(size=config.metrics.rolling_average_window)
    acc_metric = RollingAverage.create(size=config.metrics.rolling_average_window)
    dropout_rng = split(PRNGKey(0))[1]

    train_state = TrainStateWithMetrics.create(apply_fn=apply_fn, params=params, tx=tx, loss_metric=loss_metric, acc_metric=acc_metric, dropout_rng=dropout_rng)
    return train_state
