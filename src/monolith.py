import os

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_cache"
import numpy as np
from datasets import Audio, load_dataset
from transformers import WhisperFeatureExtractor
from transformers import BertTokenizer, WhisperTokenizer
import yaml
from ml_collections.config_dict import ConfigDict
from transformers import FlaxBertForSequenceClassification, FlaxWhisperForConditionalGeneration
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from flax.training.train_state import TrainState
from jax.random import PRNGKey, split
import optax
from flax.struct import PyTreeNode
from jax import jit, value_and_grad
from jax.lax import psum
from src.common.loss_and_metrics import loss_and_metrics
from typing import Dict
from jax import jit
import jax
import wandb
from tqdm.auto import tqdm
from src.data.process_dataset import strategies
import datasets
import logging
import multiprocessing as mp
import sys
import time
import transformers
from datasets import load_from_disk
from flax.training.common_utils import shard
from torch.utils.data import DataLoader
from tqdm import tqdm
from jax.nn import log_softmax
from optax import sigmoid_binary_cross_entropy
from flax import struct, traverse_util
from flax.training.common_utils import onehot




logger = logging.getLogger(__name__)






completions = ["No", "Yes"]
strategies = ["Accusation", "Call for Action", "Defense", "Evidence", "Identity Declaration", "Interrogation"]


def filter_sample(sample):
    return sample["dialogue"][-1]["target"] is not None


def create_process_sample():
    max_audio_len = 480000
    max_dialogue_lookback = 10
    max_tokens_len = 448
    sampling_rate = 16000

    prompt_format = "{dialogue}Does the final utterance conform to the strategy: '{strategy}'?\nAnswer with a single word: Yes or No.\n"

    audio = Audio(sampling_rate=sampling_rate)
    bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", truncation_side="left")
    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", truncation_side="left")

    def process_sample(sample):
        if "array" not in sample["audio"]:
            sample["audio"] = audio.decode_example(sample["audio"])
        audio_array = sample["audio"]["array"][-max_audio_len:]
        sample["input_features"] = whisper_feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]

        target = sample["dialogue"][-1]["target"]
        target_strategies = set(target.split(", ")) if target is not None else None
        dialogue = "".join(f"{d['speaker']}: {d['utterance']}\n" for d in sample["dialogue"][-max_dialogue_lookback:])

        bert_choices = []
        whisper_choices = []
        for strategy in strategies:
            label = int(strategy in target_strategies)
            prompt = prompt_format.format(dialogue=dialogue, strategy=strategy)

            bert_result = bert_tokenizer(prompt, padding="max_length", truncation=True, max_length=max_tokens_len)
            bert_choices.append(
                {
                    "strategy": strategy,
                    "label": label,
                    "input_ids": bert_result.input_ids,
                    "attention_mask": bert_result.attention_mask,
                }
            )

            whisper_result = whisper_tokenizer([prompt + completions[label]], padding="max_length", truncation=True, max_length=max_tokens_len + 1, return_length=True)
            decoder_input_ids = whisper_result.input_ids[0][:-1]
            target_tokens = whisper_result.input_ids[0][1:]
            attention_mask = whisper_result.attention_mask[0][:-1]
            loss_mask = np.zeros(max_tokens_len, dtype=np.float32)
            loss_mask[[whisper_result.length[0] - 3, whisper_result.length[0] - 2]] = 1
            whisper_choices.append(
                {
                    "strategy": strategy,
                    "label": label,
                    "decoder_input_ids": decoder_input_ids,
                    "target_tokens": target_tokens,
                    "attention_mask": attention_mask,
                    "loss_mask": loss_mask,
                }
            )

        return {"bert_choices": bert_choices, "whisper_choices": whisper_choices}

    return process_sample




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


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    level = logging.INFO if jax.process_index() == 0 else logging.ERROR
    logger.setLevel(level)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()



class RollingAverage(PyTreeNode):
    size: int
    last_element: int
    mat: jnp.ndarray

    def update(self, new_value):
        mat = self.mat.at[self.last_element].set(new_value)
        last_element = (self.last_element + 1) % mat.shape[0]
        size = jnp.where(self.size != mat.shape[0], self.size + 1, self.size)

        curr_value = mat.sum() / size
        new_value = self.replace(size=size, last_element=last_element, mat=mat)
        return curr_value, new_value

    @classmethod
    def create(cls, *, size: int):
        zero_mat = jnp.zeros(size, dtype=jnp.float32)

        rolling_average = cls(size=0, last_element=0, mat=zero_mat)
        return rolling_average



class TrainStateWithMetrics(TrainState):
    loss_metric: RollingAverage
    acc_metric: RollingAverage
    dropout_rng: PRNGKey

    def replicate(self):
        replicated = replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))
        return replicated





BERT_CONFIG = """
evaluation:
  batch_size: 64
  eval_freq: 20
  eval_steps: 5
metrics:
  rolling_average_window: 10
model:
  name: "google-bert/bert-base-cased"
training:
  b2: 0.95
  batch_size: 64
  lr: 5e-5
  total_steps: 1000
  warmup_steps: 100
  eps: 1e-6
  wd: 0.001
"""


WHISPER_CONFIG = f"""
evaluation:
  batch_size: 64
  eval_freq: 20
  eval_steps: 5
metrics:
  rolling_average_window: 10
model:
  name: "openai/whisper-small"
training:
  b2: 0.95
  batch_size: 64
  lr: 5e-5
  total_steps: 1000
  warmup_steps: 100
  eps: 1e-8
  wd: 0.01
"""


def get_config(model_name):
    if "bert" in model_name:
        return ConfigDict(yaml.safe_load(BERT_CONFIG))
    elif "whisper" in model_name:
        return ConfigDict(yaml.safe_load(WHISPER_CONFIG))
    else:
        raise Exception(f"Model name {model_name} is not supported.")

from optax import linear_schedule, join_schedules


def create_learning_rate_schedule(config):
    base_lr = float(config.training.lr)
    warmup_steps = int(config.training.warmup_steps)
    decay_steps = int(config.training.total_steps) - warmup_steps

    warmup_fn = linear_schedule(init_value=0.0, end_value=base_lr, transition_steps=warmup_steps)
    decay_fn = linear_schedule(init_value=base_lr, end_value=0.0, transition_steps=decay_steps)

    schedule_fn = join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])
    return schedule_fn






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




@jit
def whisper_train_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        outputs = state.apply_fn(**{"params": params},
                                 input_features=batch["input_features"],
                                 decoder_input_ids=batch["decoder_input_ids"],
                                 decoder_attention_mask=batch["attention_mask"], train=True)

        loss, metrics = loss_and_metrics(outputs.logits, batch["target_tokens"], batch["loss_mask"])
        preds = None
        return loss, (metrics, preds)

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, preds)), grads = grad_fn(state.params)

    grads = psum(grads, "batch")
    new_state = state.apply_gradients(grads=grads)

    loss = psum(loss, "batch")
    curr_loss, new_loss = new_state.loss_metric.update(loss)

    metrics = psum(metrics, "batch")
    acc = metrics["correct_sum"] / metrics["total_sum"]
    curr_acc, new_acc = new_state.acc_metric.update(acc)

    new_state = new_state.replace(loss_metric=new_loss, acc_metric=new_acc)

    return new_state, curr_loss, curr_acc, preds




@jit
def whisper_eval_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        outputs = state.apply_fn(**{"params": params}, input_features=batch["input_features"], decoder_input_ids=batch["decoder_input_ids"], decoder_attention_mask=batch["attention_mask"], train=True)

        loss, metrics = loss_and_metrics(outputs.logits, batch["target_tokens"], batch["loss_mask"])
        return loss, metrics

    loss, metrics = loss_fn(state.params)

    loss = psum(loss, "batch")
    metrics = psum(metrics, "batch")
    acc = metrics["correct_sum"] / metrics["total_sum"]

    return loss, acc



@jit
def bert_train_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        labels = batch["labels"]
        # jax.debug.print("🤯 labels={labels}", labels=labels)
        input_ids = batch["input_ids"]
        # jax.debug.print("🤯 input_ids={input_ids}", input_ids=input_ids[0])
        # print(batch["input_ids"].shape)
        
        outputs = state.apply_fn(**{"params": params},
                                 input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 train=True,
                                 dropout_rng=state.dropout_rng)

        # preds = jnp.argmax(outputs.logits, axis=-1)
        logits = outputs.logits[...,0]
        
        # loss, metrics = loss_and_metrics(outputs.logits, batch["labels"])
    
    
        labels = labels.astype(logits.dtype)
        log_p = jax.nn.log_sigmoid(outputs.logits)
        preds = logits>0
        
        # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
        log_not_p = jax.nn.log_sigmoid(-logits)
        loss =  -labels * log_p - (1. - labels) * log_not_p
        loss = jnp.where(jnp.isnan(loss), 0, loss)
        loss = jnp.sum(loss)
        metrics = {"correct_sum": jnp.sum(preds == labels), "total_sum": labels.shape[0]}
        # preds = None
        return loss, (preds, metrics)

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, (preds, metrics)), grads = grad_fn(state.params)


    # del preds
    grads = psum(grads, "batch")
    new_state = state.apply_gradients(grads=grads)

    loss = psum(loss, "batch")
    curr_loss, new_loss = new_state.loss_metric.update(loss)

    metrics = psum(metrics, "batch")
    acc = metrics["correct_sum"] / metrics["total_sum"]
    curr_acc, new_acc = new_state.acc_metric.update(acc)

    new_state = new_state.replace(loss_metric=new_loss, acc_metric=new_acc)

    return new_state, curr_loss, curr_acc, preds




@jit
def bert_eval_step(state: TrainStateWithMetrics, batch: Dict[str, jnp.ndarray]):
    def loss_fn(params):
        labels = batch["labels"],
        jax.debug.print("🤯 labels={labels}", labels=labels)
        input_ids = batch["input_ids"],
        jax.debug.print("🤯 input_ids={input_ids}", input_ids=input_ids[0])
        outputs = state.apply_fn(**{"params": params}, 
                                input_ids=batch["input_ids"], 
                                attention_mask=batch["attention_mask"], 
                                deterministic=True)

        # loss, metrics = loss_and_metrics(outputs.logits, batch["labels"][...,None], batch["attention_mask"])
        loss, metrics = loss_and_metrics(outputs.logits, batch["labels"][...,None])
        return loss, metrics

    loss, metrics = loss_fn(state.params)

    loss = psum(loss, "batch")
    metrics = psum(metrics, "batch")
    acc = metrics["correct_sum"] / metrics["total_sum"]

    return loss, acc



strategies_len = len(strategies)
max_audio_len = 480000
max_dialogue_lookback = 10
max_tokens_len = 448
sampling_rate = 16000
prompt_format = "{dialogue}Does the final utterance conform to the strategy: '{strategy}'?\nAnswer with a single word: Yes or No.\n"
audio = Audio(sampling_rate=sampling_rate)
import random


class BertDataCollator:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",
                                                    #    truncation_side="left"
                                                       )

    def __call__(self, features):
        prompt_list = []
        label_list = []
        strategy_list = []
        for sample in features:
            # assert filter_samples(sample)
            target = sample["dialogue"][-1]["target"]
            assert target is not None
            target_strategies = set(target.split(", "))
            dialogue = "\n".join(f"{d['speaker']}: {d['utterance']}" for d in sample["dialogue"][-max_dialogue_lookback:])
            strategy = random.choice(strategies)
            label = int(strategy in target_strategies)
            prompt = prompt_format.format(dialogue=dialogue, strategy=strategy)
            strategy_list.append(strategy)
            label_list.append(label)
            prompt_list.append(prompt)
        
        encoding = self.tokenizer.batch_encode_plus(prompt_list, padding="max_length", truncation=True, max_length=max_tokens_len, return_tensors="np")
        encoding = {**encoding}
        encoding["strategy"] = strategy_list
        encoding["labels"] = np.array(label_list, dtype=np.int32)

        return encoding



class WhisperDataCollator:
    def __init__(self):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", truncation_side="left", padding_side="left", bos_token="<|startoftranscript|>")
        
    def __call__(self, features):
        audio_list = []
        for example in features:
            if "array" not in example["audio"]:
                arr = audio.decode_example(example["audio"])
            else:
                arr = example["audio"]["array"]
            audio_list.append(arr[-max_audio_len:])
            
        # rand_indices = np.random.randint(0, strategies_len, size=len(features))
        # whisper_choices = [features[i]["whisper_choices"][rand_indices[i]] for i in range(len(features))]
        strategy_list = []
        label_list = []
        prompt_list = []
        for sample in features:
            # assert filter_samples(sample)
            target = sample["dialogue"][-1]["target"]
            assert target is not None
            target_strategies = set(target.split(", "))
            dialogue = "\n".join(f"{d['speaker']}: {d['utterance']}" for d in sample["dialogue"][-max_dialogue_lookback:])
            strategy = random.choice(strategies)
            label = int(strategy in target_strategies)
            prompt = prompt_format.format(dialogue=dialogue, strategy=strategy)  + completions[label]
            strategy_list.append(strategy)
            label_list.append(label)
            prompt_list.append(prompt)
        encoding = self.tokenizer.batch_encode_plus(prompt_list, padding="max_length", truncation=True, max_length=max_tokens_len+ 1, return_tensors="np")
        encoding = {**encoding}
        tokens = encoding.pop("input_ids")
        
        encoding["decoder_input_ids"] = tokens[:,:-1]
        encoding["target_tokens"] = tokens[:,1:]
        encoding["attention_mask"] = encoding["attention_mask"][:,:-1]
        loss_mask = np.zeros_like(encoding["attention_mask"], dtype=np.int32)
        loss_mask[:,-1] = 1
        encoding["loss_mask"] = loss_mask
        encoding["strategy"] = strategy_list
        encoding["labels"] = np.array(label_list, dtype=np.int32)
        encoding["input_features"] = self.feature_extractor(np.array(audio_list), sampling_rate=sampling_rate).input_features

        return encoding





class TrainStateWithMetrics(TrainState):
    loss_metric: RollingAverage
    acc_metric: RollingAverage
    dropout_rng: PRNGKey

    def replicate(self):
        replicated = replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))
        return replicated

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
    loss_metric = RollingAverage.create(size=config.metrics.rolling_average_window)
    acc_metric = RollingAverage.create(size=config.metrics.rolling_average_window)
    dropout_rng = split(PRNGKey(0))[1]

    train_state = TrainStateWithMetrics.create(apply_fn=apply_fn, params=params, tx=tx, loss_metric=loss_metric, acc_metric=acc_metric, dropout_rng=dropout_rng)
    return train_state


def worker_init_fn(worker_id):
    np.random.seed(worker_id + int(time.time()))

from transformers import AutoConfig


from collections import namedtuple


STEPS = namedtuple("STEPS", ["train_step", "eval_step"])

bert_steps = STEPS(bert_train_step, bert_eval_step)
whisper_steps = STEPS(whisper_train_step, whisper_eval_step)

def get_data_collator(model_name):
    if "bert" in model_name:
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
        model = FlaxBertForSequenceClassification.from_pretrained(model_name, from_pt=True, config=config)
        return BertDataCollator(), bert_steps, model
    else:
        assert "whisper" in model_name
        model = FlaxWhisperForConditionalGeneration.from_pretrained(model_name, from_pt=True)
        return WhisperDataCollator(), whisper_steps, model



def filter_samples(sample):
    target = sample["dialogue"][-1]["target"]
    if target is None:
        return False
    target_strategies = set(target.split(", "))
    if len(target_strategies) == 0:
        return False
    return True


def train(model_name):
    setup_logging()
    mp.set_start_method("spawn", force=True)
    config = get_config(model_name)
    # dataset = load_from_disk("/dev/shm/hf_cache/werewolf_data")
    num_proc = os.cpu_count() // 2

    dataset = load_dataset("iohadrubin/werewolf_dialogue_data_10sec", num_proc=num_proc)
    # new_dataset = datasets.DatasetDict()
    # for split in ["train",
    #             #   "validation","test"
    #               ]:
    #     new_dataset[split] = dataset[split].select(range(300))
    # dataset = new_dataset
    
    dataset = dataset.filter(filter_samples, num_proc=num_proc)

    # process_sample = create_process_sample()
    # dataset = dataset.map(process_sample, num_proc=num_proc)

    dataset = dataset.shuffle(seed=42)
    dataset = dataset.flatten_indices(num_proc=num_proc)

    device_count = jax.device_count()
    train_batch_size = 8 * device_count
    num_workers = os.cpu_count() // 2
    prefetch_factor = 4
    data_collator, steps, model = get_data_collator(model_name)

    common_kwargs = dict(
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    train_loader = DataLoader(dataset["train"], batch_size=train_batch_size, shuffle=True, **common_kwargs)
    num_epochs = 2
    worker_id = jax.process_index()
    if worker_id == 0:
        wandb.init(project="whisper_werewolf", config=config.to_dict())

    lr_schedule = create_learning_rate_schedule(config)

    state = create_train_state(config, model, lr_schedule)
    state = state.replicate()

    p_train_step = jax.pmap(steps.train_step, "batch", donate_argnums=(0,))

    pbar = tqdm(range(config.training.total_steps), desc="Training")
    epochs = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epochs:
        for step, batch in zip(pbar, train_loader):
            print(jax.tree.map(np.shape, batch))
            epoch = batch.pop("epoch", 0)
            strategy = batch.pop("strategy")
            batch = shard(batch)

            state, curr_loss, curr_acc, preds = p_train_step(state, batch)
            # if step > 10:
            #     assert False
            #     print(f"{preds=}")
            #     assert preds is not None
            curr_loss = curr_loss.mean().item()
            curr_acc = curr_acc.mean().item()

            pbar.set_description(f"Loss: {curr_loss:.4f}, Acc: {curr_acc:.4f}")
            metrics = {"step": step, "loss": float(curr_loss), "accuracy": float(curr_acc), "lr": float(lr_schedule(step)), "epoch": epoch}
            print(metrics)

            if worker_id == 0:
                wandb.log(metrics)

    if worker_id == 0:
        wandb.finish()
        # params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
        # model.save_pretrained(os.path.join(os.path.dirname(__file__), f"model/{model_name}"), params=params)


import fire

# python3.10 -m src.monolith --model_name=openai/whisper-small
# python3.10 -m src.monolith --model_name=google-bert/bert-base-cased
if __name__ == "__main__":
    fire.Fire(train)
