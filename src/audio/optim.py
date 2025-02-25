import optax


def create_learning_rate_schedule(config):
    """
    Create a linear warmup + linear decay schedule using optax.
    We'll replicate the get_linear_schedule_with_warmup logic from the example.
    """
    warmup_steps = int(config.training.warmup_steps)
    total_steps = int(config.training.total_steps)
    base_lr = float(config.training.lr)

    # Phase 1: Warmup from 0 to base_lr over warmup_steps
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )

    # Phase 2: Decay from base_lr to 0 over the remaining steps
    decay_steps = max(1, total_steps - warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=base_lr,
        end_value=0.0,
        transition_steps=decay_steps,
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps],
    )
    return schedule_fn
