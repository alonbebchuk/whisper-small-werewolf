import evaluate
import jax
import numpy as np

from src.new.datasets import strategies


def get_strategy_metrics(predictions, labels, strategy_id):
    predictions_host = np.array(jax.device_get(predictions))
    labels_host = np.array(jax.device_get(labels))
    accuracy_metric = evaluate.load("accuracy")
    strategy_metrics = {}
    for strat_id, strat in enumerate(strategies):
        indices = [i for i, s_id in enumerate(strategy_id) if s_id == strat_id]
        predictions_subset = [int(predictions_host[i]) for i in indices]
        labelss_subset = [int(labels_host[i]) for i in indices]
        accuracy = accuracy_metric.compute(predictions=predictions_subset, references=labelss_subset)["accuracy"]
        total = len(indices)
        correct = int(accuracy * total)
        strategy_metrics[f"{strat}_accuracy"] = accuracy
        strategy_metrics[f"{strat}_correct"] = correct
        strategy_metrics[f"{strat}_total"] = total
    return strategy_metrics


def create_train_step(state, batch, apply_fn_kwargs, loss_and_grad_fn):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    labels = batch.pop("labels")

    def loss_fn(params):
        outputs = state.apply_fn(**apply_fn_kwargs, params=params, dropout_rng=dropout_rng, train=True)
        return state.loss_fn(outputs.logits, labels)

    loss, grad = loss_and_grad_fn(loss_fn)
    new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)
    metrics = {"loss": loss, "learning_rate": state.learning_rate_fn(state.step)}
    return new_state, metrics


def create_eval_step(state, batch, apply_fn_kwargs, loss_fn):
    labels = batch.pop("labels")
    outputs = state.apply_fn(**apply_fn_kwargs, params=state.params, train=False)
    loss = loss_fn(outputs.logits, labels)
    # predictions = state.logits_fn(logits)
    # strategy_metrics = get_strategy_metrics(predictions, labels, batch["strategy_id"])
    # metrics = {"loss": loss}.update(strategy_metrics)
    metrics = {"loss": loss}
    return metrics


def get_bert_train_step(state, batch):
    apply_fn_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }

    def loss_and_grad_fn(loss_fn):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        loss = jax.lax.pmean(loss, "batch")
        grad = jax.lax.pmean(grad, "batch")
        return loss, grad

    return create_train_step(state, batch, apply_fn_kwargs, loss_and_grad_fn)


def get_whisper_train_step(state, batch):
    apply_fn_kwargs = {
        "input_features": batch["input_features"],
        "decoder_input_ids": batch["decoder_input_ids"],
    }

    def loss_and_grad_fn(loss_fn):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, num_labels), grad = grad_fn(state.params)
        num_labels = jax.lax.psum(num_labels, "batch")
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)
        grad = jax.lax.psum(grad, "batch")
        grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
        return loss, grad

    return create_train_step(state, batch, apply_fn_kwargs, loss_and_grad_fn)


def get_bert_eval_step(state, batch):
    apply_fn_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }

    def loss_fn(logits, labels):
        loss = state.loss_fn(logits, labels)
        loss = jax.lax.pmean(loss, "batch")
        return loss

    return create_eval_step(state, batch, apply_fn_kwargs, loss_fn)


def get_whisper_eval_step(state, batch):
    apply_fn_kwargs = {
        "input_features": batch["input_features"],
        "decoder_input_ids": batch["decoder_input_ids"],
    }

    def loss_fn(logits, labels):
        loss, num_labels = state.loss_fn(logits, labels)
        num_labels = jax.lax.psum(num_labels, "batch")
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)
        return loss

    return create_eval_step(state, batch, apply_fn_kwargs, loss_fn)


_steps = None


def get_steps(model_name):
    global _steps
    if _steps is None:
        if model_name == "bert":
            _steps = (get_bert_train_step, get_bert_eval_step)
        elif model_name == "whisper":
            _steps = (get_whisper_train_step, get_whisper_eval_step)
        else:
            raise Exception(f"Model name {model_name} is not supported.")
    return _steps
