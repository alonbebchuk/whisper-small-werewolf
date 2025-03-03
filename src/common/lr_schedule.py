from optax import linear_schedule, join_schedules


def create_learning_rate_schedule(config):
    base_lr = float(config.training.lr)
    warmup_steps = int(config.training.warmup_steps)
    decay_steps = int(config.training.total_steps) - warmup_steps

    warmup_fn = linear_schedule(init_value=0.0, end_value=base_lr, transition_steps=warmup_steps)
    decay_fn = linear_schedule(init_value=base_lr, end_value=0.0, transition_steps=decay_steps)

    schedule_fn = join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])
    return schedule_fn
