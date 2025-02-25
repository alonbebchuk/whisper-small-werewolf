
# python3.10 -m  src.train
# python3.10 -m  pip install torchvision==0.16.0
# python3.10 -m  pip install  evaluate==0.4.3
#python3.10 -m  pip install --upgrade datasets
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import time
import yaml
import wandb
import jax
import jax.numpy as jnp
import optax
import numpy as np


from typing import Dict, Any
from tqdm.auto import tqdm
from flax import linen as nn

from flax.training import train_state



from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
# For dataset loading/processing
import torch
from torch.utils.data import DataLoader

import evaluate
from datasets import load_dataset


# Hugging Face Transformers in Flax
from flax import jax_utils

from src.models.vit import FlaxViTForImageClassification, ViTConfig
from src.data import create_collate_fn, create_preprocess_transforms
from datasets import IterableDataset
from src.rolling_avg import RollingAverage
from src.optim import create_learning_rate_schedule

from ml_collections import config_dict
import yaml

# name: "timm/imagenet-1k-wds"
CONFIG = """
model:
  rubin_parametrization: True

dataset:
    name: "cifar100"
  

figure_size:
  width: 10
  height: 5
metrics:
  rolling_average_window: 20
training:
  total_steps: 100000
  warmup_steps: 10000
  lr: 5e-5
  wd: 0.01
  b2: 0.95
  batch_size: 64
"""


def get_config():
    """
    Load config from the above YAML string into a ConfigDict.
    """
    config_dict_raw = yaml.safe_load(CONFIG)
    return config_dict.ConfigDict(config_dict_raw)


# --------------------------------------------------------------------------------
# Dataset / Preprocessing
# --------------------------------------------------------------------------------
def load_dataset_and_metrics(config):
    metric = evaluate.load("accuracy")
    name = config.dataset.name
    dataset = load_dataset(name, streaming="imagenet" in name)

    # If needed, rename column from "img" -> "image" to match transforms
    # but some HF datasets for CIFAR10 have "img" vs "image" depending on version
    # Here we do a safety check:
    if "image" not in dataset["train"].column_names:
        if "img" in dataset["train"].column_names:
            dataset = dataset.rename_column("img", "image")
        if "jpg" in dataset["train"].column_names:
            dataset = dataset.rename_column("jpg", "image")
    if "imagenet" in name:
        label2id, id2label = {}, {}
        for i in range(1000):
            label2id[i] = i
            id2label[i] = i
        dataset = dataset.rename_column("cls", "label")
        
    else:
        if "label" in dataset["train"].features:
            labels = dataset["train"].features["label"].names
            label2id, id2label = {}, {}
            for i, label in enumerate(labels):
                label2id[label] = i
                id2label[i] = label
        elif "fine_label" in dataset["train"].features:
            labels = dataset["train"].features["fine_label"].names
            label2id, id2label = {}, {}
            for i, label in enumerate(labels):
                label2id[label] = i
                id2label[i] = label
            dataset = dataset.rename_column("fine_label", "label")
    return dataset, metric, label2id, id2label



class TrainStateWithMetrics(train_state.TrainState):
    """
    Extends the basic Flax TrainState with rolling metrics for loss & accuracy.
    """
    loss_metric: RollingAverage
    acc_metric: RollingAverage
    dropout_rng: jax.random.PRNGKey
    
    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))




class DataStream:
    def __init__(self, config, splits=["train", "validation","test"]):
        super().__init__()
        # self.dataloader = dataloader
        self.config = config
        dataset, _, label2id, id2label = load_dataset_and_metrics(config)
        self._num_labels = len(label2id)
        self._label2id = label2id
        self._id2label = id2label
        collate_fn = create_collate_fn()  # We'll convert to jax arrays ourselves

        batch_size = config.training.batch_size
        # 4. Create transforms & set them on HF dataset
        model_checkpoint = "microsoft/swin-tiny-patch4-window7-224"  # same as PyTorch example
        preprocess_train, preprocess_val, _ = create_preprocess_transforms(model_checkpoint)
        if "validation" in dataset:
            train_ds = dataset["train"]
            val_ds = dataset["validation"]
        else:
            splits = dataset["train"].train_test_split(test_size=0.1, seed=42)
            train_ds = splits["train"]
            val_ds = splits["test"]

        
        train_ds  = train_ds.shard(num_shards=jax.process_count(), index=jax.process_index())
        if isinstance(train_ds, IterableDataset):
            train_ds = train_ds.map(preprocess_train, batched=True, batch_size=batch_size)
            kwargs = dict(num_workers=20,prefetch_factor=10)
        else:
            kwargs = dict(shuffle=True, drop_last=True)
            train_ds.set_transform(preprocess_train)
        self.train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=batch_size, **kwargs)
        
        val_ds  = val_ds.shard(num_shards=jax.process_count(), index=jax.process_index())
        if isinstance(val_ds, IterableDataset):
            val_ds = val_ds.map(preprocess_val, batched=True, batch_size=batch_size)
            kwargs = {}
        else:
            val_ds.set_transform(preprocess_val)
            kwargs = dict(shuffle=False, drop_last=False)
        self.dev_dl = DataLoader(val_ds, collate_fn=collate_fn, batch_size=batch_size, **kwargs )

    @property
    def num_labels(self):
        return self._num_labels
    @property
    def label2id(self):
        return self._label2id
    @property
    def id2label(self):
        return self._id2label
    
    
    def generate(self, dataloader):
        itr = iter(dataloader)
        epoch = 0
        while True:
            try:
                batch = next(itr)
            except StopIteration:
                # Re-start DataLoader if exhausted
                itr = iter(dataloader)
                batch = next(itr)
                epoch += 1
            batch = shard(batch)
            batch["epoch"] = epoch
            yield batch


    def train_iter(self):
        return self.generate(self.train_dl)

    def validation_iter(self):
        if self.dev_dl is None:
            raise ValueError("No validation data found")
        return self.generate(self.dev_dl)




def create_train_state(config, model, input_shape):

    
    rng = jax.random.PRNGKey(0)
    rng, dropout_rng = jax.random.split(rng)
    

    params = model.init_weights(rng,input_shape=input_shape)

    # Create learning rate schedule and optimizer
    lr_schedule = create_learning_rate_schedule(config)
    tx = optax.adamw(lr_schedule, weight_decay=config.training.wd, b2=config.training.b2)

    return TrainStateWithMetrics.create(
        apply_fn=model.__call__,
        params=params,
        tx=tx,
        loss_metric=RollingAverage.create(size=config.metrics.rolling_average_window),
        acc_metric=RollingAverage.create(size=config.metrics.rolling_average_window),
        dropout_rng=dropout_rng,
    )



@jax.jit
def train_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        outputs = state.apply_fn(
            **{"params": params},
            pixel_values=batch["pixel_values"],
            train=True,  # ensure model is in train mode
        )
        logits = outputs.logits  # [batch, num_labels]
        one_hot = jax.nn.one_hot(batch["labels"], num_classes=logits.shape[-1])
        unnorm_loss =  optax.softmax_cross_entropy(logits, one_hot).sum()
        
        return unnorm_loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (unnorm_loss, logits), grads = grad_fn(state.params)
    grads = jax.lax.psum(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    
    
    predictions = jnp.argmax(logits, axis=-1) == batch["labels"]
    is_correct = jnp.sum(predictions)
    
    total_n_examples = jax.lax.psum(logits.shape[0], "batch")
    total_is_correct = jax.lax.psum(is_correct, "batch")
    total_loss = jax.lax.psum(unnorm_loss, "batch")
    acc = total_is_correct / total_n_examples
    loss = total_loss / total_n_examples

    # Update rolling average metrics
    curr_loss, new_loss_metric = new_state.loss_metric.update(loss)
    curr_acc, new_acc_metric = new_state.acc_metric.update(acc)

    # Replace the old metrics with updated ones
    new_state = new_state.replace(loss_metric=new_loss_metric, acc_metric=new_acc_metric)

    return new_state, curr_loss, curr_acc, total_n_examples

@jax.jit
def eval_step(state: TrainStateWithMetrics,
              batch: Dict[str, jnp.ndarray]):
    outputs = state.apply_fn(
        **{"params": state.params},
        pixel_values=batch["pixel_values"],
        train=True,  # ensure model is in train mode
    )
    logits = outputs.logits  # [batch, num_labels]
    one_hot = jax.nn.one_hot(batch["labels"], num_classes=logits.shape[-1])
    unnorm_loss =  optax.softmax_cross_entropy(logits, one_hot).sum()
    predictions = jnp.argmax(logits, axis=-1) == batch["labels"]
    is_correct = jnp.sum(predictions)
    
    total_n_examples = jax.lax.psum(logits.shape[0], "batch")
    total_is_correct = jax.lax.psum(is_correct, "batch")
    total_loss = jax.lax.psum(unnorm_loss, "batch")
    acc = total_is_correct / total_n_examples
    loss = total_loss / total_n_examples


    return loss, acc



def create_model(config, stream):
    model_config = ViTConfig(
        num_labels=stream.num_labels,
        label2id=stream.label2id,
        id2label=stream.id2label,
        # ignoring mismatched sizes for demonstration, as in the PyTorch code
        ignore_mismatched_sizes=True,
        **dict(config.model),
    )
    model = FlaxViTForImageClassification(model_config)
    input_shape = (1, model_config.image_size, model_config.image_size, model_config.num_channels)
    return model, model_config, input_shape


def main():
    config = get_config()
    worker_id = jax.process_index()
    if worker_id==0:
        wandb.init(project=f"vit_jax_{config.dataset.name.split('/')[-1]}", config=config.to_dict())

    
    stream = DataStream(config)

 
    lr_schedule = create_learning_rate_schedule(config)
    
    
    model, _, input_shape = create_model(config, stream)
    state = create_train_state(config, model, input_shape)
    state = state.replicate()
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=tuple())
    

    total_steps = config.training.total_steps
    pbar = tqdm(range(total_steps), desc="Training")
    eval_freq = 2000
    eval_steps = 5
    eval_counter = eval_freq
    seen_examples = 0
    for step, batch in zip(pbar, stream.train_iter()):
        # Single train step
        epoch = batch.pop("epoch", 0)
        
        state, curr_loss, curr_acc, total_n_examples = p_train_step(state, batch)
        total_n_examples = int(total_n_examples[0])
        seen_examples += total_n_examples
        curr_loss = curr_loss.mean().item()
        curr_acc = curr_acc.mean().item()
        

        pbar.set_description(f"Loss: {curr_loss:.4f}, Acc: {curr_acc:.4f}")
        metrics = {
                "step": step,
                "loss": float(curr_loss),
                "accuracy": float(curr_acc),
                "lr": float(lr_schedule(step)),
                "epoch": epoch,
                "seen_examples": seen_examples,
            }
        
        eval_counter -= 1
        if eval_counter==0:
            eval_counter = eval_freq
            for i, dev_batch in enumerate(stream.validation_iter()):
                if i>=eval_steps:
                    break
                dev_batch.pop("epoch", 0)
                curr_loss, curr_acc = p_eval_step(state, dev_batch)
                curr_loss = curr_loss.mean().item()
                curr_acc = curr_acc.mean().item()
                
            if worker_id==0:
                wandb.log({"eval_loss": curr_loss, "eval_accuracy": curr_acc, "epoch": epoch, "seen_examples": seen_examples,
                           "step": step})

        # Log to wandb
        if worker_id==0:
            wandb.log(metrics)
    if worker_id==0:
        wandb.finish()


if __name__ == "__main__":
    main()
