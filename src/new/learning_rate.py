import optax

_learning_rate_fn = None


def get_learning_rate_fn(training_args):
    global _learning_rate_fn
    if _learning_rate_fn is None:
        warmup_fn = optax.linear_schedule(init_value=0.0, end_value=training_args.learning_rate, transition_steps=training_args.num_warmup_steps)
        decay_fn = optax.linear_schedule(init_value=training_args.learning_rate, end_value=0, transition_steps=training_args.num_train_steps - training_args.num_warmup_steps)
        _learning_rate_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[training_args.num_warmup_steps])
    return _learning_rate_fn
