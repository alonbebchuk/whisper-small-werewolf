#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import datasets
import evaluate
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct, traverse_util
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import HfApi
from src.bert.utils_qa import postprocess_qa_predictions
from src.common.config import strategies
from tqdm import tqdm


import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    FlaxAutoModelForQuestionAnswering,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    is_tensorboard_available,
)
from transformers.utils import check_min_version, send_example_telemetry


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.50.0.dev0")

Array = Any
Dataset = datasets.arrow_dataset.Dataset
PRNGKey = Any


# region Arguments
@dataclass
class TrainingArguments:
    output_dir: str = field(metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    overwrite_output_dir: bool = field(default=False, metadata={"help": ("Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory.")})
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."})
    max_seq_length: int = field(default=384, metadata={"help": ("The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated, sequences shorter will be padded.")})
    doc_stride: int = field(default=128, metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."})
    n_best_size: int = field(default=20, metadata={"help": "The total number of n-best predictions to generate when looking for an answer."})
    max_answer_length: int = field(default=30, metadata={"help": ("The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")})


# endregion


# region Create a train state
def create_train_state(
    model: FlaxAutoModelForQuestionAnswering,
    learning_rate_fn: Callable[[int], float],
    num_labels: int,
    training_args: TrainingArguments,
) -> train_state.TrainState:
    """Create initial training state."""

    class TrainState(train_state.TrainState):
        """Train state with an Optax optimizer.

        The two functions below differ depending on whether the task is classification
        or regression.

        Args:
          logits_fn: Applied to last layer to obtain the logits.
          loss_fn: Function to compute the loss.
        """

        logits_fn: Callable = struct.field(pytree_node=False)
        loss_fn: Callable = struct.field(pytree_node=False)

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {layer[-2:] for layer_norm_name in layer_norm_candidates for layer in flat_params.keys() if layer_norm_name in "".join(layer).lower()}
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    def cross_entropy_loss(logits, labels):
        start_loss = optax.softmax_cross_entropy(logits[0], onehot(labels[0], num_classes=num_labels))
        end_loss = optax.softmax_cross_entropy(logits[1], onehot(labels[1], num_classes=num_labels))
        xentropy = (start_loss + end_loss) / 2.0
        return jnp.mean(xentropy)

    return TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
        logits_fn=lambda logits: logits,
        loss_fn=cross_entropy_loss,
    )


# endregion


# region Create learning rate function
def create_learning_rate_fn(train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float) -> Callable[[int], jnp.ndarray]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


# endregion


# region train data iterator
def train_data_collator(rng: PRNGKey, dataset: Dataset, batch_size: int):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: np.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch


# endregion


# region eval data iterator
def eval_data_collator(dataset: Dataset, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`. Sharding handled by `pad_shard_unpad` in the eval loop."""
    batch_idx = np.arange(len(dataset))

    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        # Ignore `offset_mapping` to avoid numpy/JAX array conversion issue.
        batch = {k: np.array(v) for k, v in batch.items() if k != "offset_mapping"}

        yield batch


# endregion


def main():
    # region Argument parsing
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_qa", model_args, data_args, framework="flax")
    # endregion

    # region Logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # endregion

    # region Load Data
    def get_strategy_dataset_name(strategy):
        strategy_dataset_name = f"alonbeb/werewolf_{strategy.replace(' ', '-')}_data"
        return strategy_dataset_name

    def strategy_to_question(strategy):
        question = f"Does the last utterance fit the strategy category '{strategy}'?"
        return question

    def dialogue_to_context(dialogue):
        context_prefix = "Respond with a single word: Yes or No.\n"
        context_dialogue = "\n".join(f"{x['speaker']}: {x['utterance']}" for x in dialogue[:-2])
        if len(context_dialogue) > 256:
            context_dialogue = context_dialogue[-256:]
        return context_prefix + context_dialogue

    def strategy_and_dialogue_to_answer(strategy, dialogue):
        answer = {"text": ["Yes"], "answer_start": [28]} if strategy in dialogue[-1]["target"].split(", ") else {"text": ["No"], "answer_start": [35]}
        return answer
    
    def file_name_and_idx_to_id(file_name, idx):
        parts = file_name.split("_")
        file_num = parts[1]
        game = parts[2]
        id = f"File{file_num}-{game}-Idx{idx}"
        return id

    def prepare_features(examples):
        examples["question"] = [strategy_to_question(strategy) for strategy in examples["strategy"]]
        examples["context"] = [dialogue_to_context(dialogue) for dialogue in examples["dialogue"]]
        examples["answers"] = [strategy_and_dialogue_to_answer(strategy, dialogue) for (strategy, dialogue) in zip(examples["strategy"], examples["dialogue"])]
        examples["id"] = [file_name_and_idx_to_id(file_name, idx) for (file_name, idx) in zip(examples["file_name"], examples["idx"])]
        return examples

    raw_datasets = datasets.load_dataset(get_strategy_dataset_name(strategies[0]))
    raw_datasets = raw_datasets.map(prepare_features, batched=True, num_proc=10)
    # raw_datasets = datasets.DatasetDict()

    # strategy_datasets = [datasets.load_dataset(get_strategy_dataset_name(strategy), num_proc=10) for strategy in strategies]
    # raw_datasets["train"] = datasets.concatenate_datasets([raw_dataset["train"] for raw_dataset in strategy_datasets])
    # raw_datasets["validation"] = datasets.concatenate_datasets([raw_dataset["validation"] for raw_dataset in strategy_datasets])
    # raw_datasets["test"] = datasets.concatenate_datasets([raw_dataset["test"] for raw_dataset in strategy_datasets])
    # endregion

    # region Load pretrained model and tokenizer
    #
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    # endregion

    # region Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("This example script only works for models that have a fast tokenizer. Checkout the big table of models at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this requirement")
    # endregion

    # region Preprocessing the datasets
    # Preprocessing is slightly different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    processed_raw_datasets = {}
    if training_args.do_train:
        # Create train feature from dataset
        train_dataset = raw_datasets["train"].shard(num_shards=jax.process_count(), index=jax.process_index())
        train_dataset = train_dataset.map(prepare_train_features, batched=True, remove_columns=column_names, num_proc=10)
        # train_dataset = datasets.Dataset.from_generator(lambda: (yield from train_dataset), features=train_dataset.features)
        processed_raw_datasets["train"] = train_dataset

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [(o if sequence_ids[k] == context_index else None) for k, o in enumerate(tokenized_examples["offset_mapping"][i])]

        return tokenized_examples

    if training_args.do_eval:
        eval_examples = raw_datasets["validation"].shard(num_shards=jax.process_count(), index=jax.process_index())
        # Validation Feature Creation
        eval_dataset = eval_examples.map(prepare_validation_features, batched=True, remove_columns=column_names, num_proc=10)
        # eval_dataset = datasets.Dataset.from_generator(lambda: (yield from eval_dataset), features=eval_dataset.features)
        processed_raw_datasets["validation"] = eval_dataset

    if training_args.do_predict:
        predict_examples = raw_datasets["test"].shard(num_shards=jax.process_count(), index=jax.process_index())
        # Predict Feature Creation
        predict_dataset = predict_examples.map(prepare_validation_features, batched=True, remove_columns=column_names, num_proc=10)
        # predict_dataset = datasets.Dataset.from_generator(lambda: (yield from predict_dataset), features=predict_dataset.features)
        processed_raw_datasets["test"] = predict_dataset
    # endregion

    # region Metrics and Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs of the model.
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # endregion

    # region Training steps and logging init
    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Define a summary writer
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(training_args.output_dir)
            summary_writer.hparams({**training_args.to_dict(), **vars(model_args), **vars(data_args)})
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(f"Unable to display metrics through TensorBoard because some package are not installed: {ie}")
    else:
        logger.warning("Unable to display metrics through TensorBoard because the package is not installed: Please run pip install tensorboard to enable.")

    def write_train_metric(summary_writer, train_metrics, train_time, step):
        summary_writer.scalar("train_time", train_time, step)

        train_metrics = get_metrics(train_metrics)
        for key, vals in train_metrics.items():
            tag = f"train_{key}"
            for i, val in enumerate(vals):
                summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    def write_eval_metric(summary_writer, eval_metrics, step):
        for metric_name, value in eval_metrics.items():
            summary_writer.scalar(f"eval_{metric_name}", value, step)

    num_epochs = int(training_args.num_train_epochs)
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    train_batch_size = int(training_args.per_device_train_batch_size) * jax.local_device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.local_device_count()
    # endregion

    # region Load model
    model = FlaxAutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, "float32"))

    learning_rate_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    state = create_train_state(model, learning_rate_fn, num_labels=max_seq_length, training_args=training_args)
    # endregion

    # region Define train step functions
    def train_step(state: train_state.TrainState, batch: Dict[str, Array], dropout_rng: PRNGKey) -> Tuple[train_state.TrainState, float]:
        """Trains model with an optimizer (both in `state`) on `batch`, returning a pair `(new_state, loss)`."""
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        start_positions = batch.pop("start_positions")
        end_positions = batch.pop("end_positions")
        targets = (start_positions, end_positions)

        def loss_fn(params):
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)
            loss = state.loss_fn(logits, targets)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)
        metrics = jax.lax.pmean({"loss": loss, "learning_rate": learning_rate_fn(state.step)}, axis_name="batch")
        return new_state, metrics, new_dropout_rng

    p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))
    # endregion

    # region Define eval step functions
    def eval_step(state, batch):
        logits = state.apply_fn(**batch, params=state.params, train=False)
        return state.logits_fn(logits)

    p_eval_step = jax.pmap(eval_step, axis_name="batch")
    # endregion

    # region Define train and eval loop
    logger.info(f"===== Starting training ({num_epochs} epochs) =====")
    train_time = 0

    # make sure weights are replicated on each device
    state = replicate(state)

    train_time = 0
    step_per_epoch = len(train_dataset) // train_batch_size
    total_steps = step_per_epoch * num_epochs
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        train_start = time.time()
        train_metrics = []

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # train
        for step, batch in enumerate(
            tqdm(
                train_data_collator(input_rng, train_dataset, train_batch_size),
                total=step_per_epoch,
                desc="Training...",
                position=1,
            ),
            1,
        ):
            state, train_metric, dropout_rngs = p_train_step(state, batch, dropout_rngs)
            train_metrics.append(train_metric)

            cur_step = epoch * step_per_epoch + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                # Save metrics
                train_metric = unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                epochs.write(f"Step... ({cur_step}/{total_steps} | Training Loss: {train_metric['loss']}, Learning Rate:" f" {train_metric['learning_rate']})")

                train_metrics = []

            if training_args.do_eval and (cur_step % training_args.eval_steps == 0 or cur_step % step_per_epoch == 0) and cur_step > 0:
                eval_metrics = {}
                all_start_logits = []
                all_end_logits = []
                # evaluate
                for batch in tqdm(
                    eval_data_collator(eval_dataset, eval_batch_size),
                    total=math.ceil(len(eval_dataset) / eval_batch_size),
                    desc="Evaluating ...",
                    position=2,
                ):
                    _ = batch.pop("example_id")
                    predictions = pad_shard_unpad(p_eval_step)(state, batch, min_device_batch=per_device_eval_batch_size)
                    start_logits = np.array(predictions[0])
                    end_logits = np.array(predictions[1])
                    all_start_logits.append(start_logits)
                    all_end_logits.append(end_logits)

                max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

                # concatenate the numpy array
                start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
                end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

                # delete the list of numpy arrays
                del all_start_logits
                del all_end_logits
                outputs_numpy = (start_logits_concat, end_logits_concat)
                prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
                eval_metrics = compute_metrics(prediction)

                logger.info(f"Step... ({cur_step}/{total_steps} | Evaluation metrics: {eval_metrics})")

                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)

            if (cur_step % training_args.save_steps == 0 and cur_step > 0) or (cur_step == total_steps):
                # save checkpoint after each epoch and push checkpoint to the hub
                if jax.process_index() == 0:
                    params = jax.device_get(unreplicate(state.params))
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    if training_args.push_to_hub:
                        # Create repo and retrieve repo_id
                        api = HfApi()
                        repo_id = api.create_repo("bert-werewolf", exist_ok=True).repo_id
                        api.upload_folder(commit_message=f"Saving weights and logs of step {cur_step}", folder_path=training_args.output_dir, repo_id=repo_id, repo_type="model")
        epochs.desc = f"Epoch ... {epoch + 1}/{num_epochs}"
    # endregion

    # Eval after training
    if training_args.do_eval:
        eval_metrics = {}
        all_start_logits = []
        all_end_logits = []

        eval_loader = eval_data_collator(eval_dataset, eval_batch_size)
        for batch in tqdm(eval_loader, total=math.ceil(len(eval_dataset) / eval_batch_size), desc="Evaluating ...", position=2):
            _ = batch.pop("example_id")
            predictions = pad_shard_unpad(p_eval_step)(state, batch, min_device_batch=per_device_eval_batch_size)
            start_logits = np.array(predictions[0])
            end_logits = np.array(predictions[1])
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits
        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
        eval_metrics = compute_metrics(prediction)

        if jax.process_index() == 0:
            eval_metrics = {f"eval_{metric_name}": value for metric_name, value in eval_metrics.items()}
            path = os.path.join(training_args.output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
