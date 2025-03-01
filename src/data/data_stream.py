import jax
import numpy as np

from datasets import load_dataset, Audio
from flax.training.common_utils import shard
from src.common.config import get_strategy_dataset_name
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


def create_process_sample_fn(config):
    tokenizer = WhisperTokenizer.from_pretrained(config.model.name, bos_token=config.model.bos_token)

    def process_sample(sample):
        tokens = tokenizer.encode(sample["prompt"] + sample["completion"], add_special_tokens=False)
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

        slice_len = len(tokens) - 1
        padding_len = config.model.max_len - slice_len
        prompt_len = len(tokenizer.encode(sample["prompt"], add_special_tokens=False))

        input_tokens = tokens[:-1] + [tokenizer.pad_token_id] * padding_len
        target_tokens = tokens[1:] + [tokenizer.pad_token_id] * padding_len
        attention_mask = [1] * slice_len + [0] * padding_len
        loss_mask = [0.0] * prompt_len + [1.0] * (slice_len - prompt_len) + [0.0] * padding_len

        result = {
            "input_tokens": np.array(input_tokens, dtype=np.int32),
            "target_tokens": np.array(target_tokens, dtype=np.int32),
            "loss_mask": np.array(loss_mask, dtype=np.float32),
            "attention_mask": np.array(attention_mask, dtype=np.int32)
        }
        return result

    return process_sample


def create_collate_fn(config):
    audio = Audio()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model.name)
    processor = WhisperProcessor.from_pretrained(config.model.name)

    def collate(batch):
        audio_arrays = [audio.decode_example(x)["array"] for x in batch["audio"]]

        input_features = feature_extractor(audio_arrays, sampling_rate=config.model.sampling_rate).input_features
        input_features = processor.feature_extractor.pad([{"input_features": x} for x in list(input_features)], return_tensors="np").input_features

        result = {
            "input_features": input_features,
            "decoder_input_ids": np.array(batch["input_tokens"], dtype=np.int32),
            "target_tokens": np.array(batch["target_tokens"], dtype=np.int32),
            "loss_mask": np.array(batch["loss_mask"], dtype=np.float32),
            "attention_mask": np.array(batch["attention_mask"], dtype=np.int32)
        }
        return result

    return collate


class DataStream:
    def __init__(self, config, strategy):
        super().__init__()

        dataset = load_dataset(get_strategy_dataset_name(config, strategy), streaming=True)
        train_ds = dataset["train"]
        val_ds = dataset["validation"]

        process_count, process_index = jax.process_count(), jax.process_index()
        train_ds = train_ds.shard(num_shards=process_count, index=process_index)
        val_ds = val_ds.shard(num_shards=process_count, index=process_index)

        process_sample_fn = create_process_sample_fn(config)
        train_ds = train_ds.map(process_sample_fn)
        val_ds = val_ds.map(process_sample_fn)

        collate_fn = create_collate_fn(config)
        self.train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=config.training.batch_size, num_workers=20, prefetch_factor=10)
        self.val_dl = DataLoader(val_ds, collate_fn=collate_fn, batch_size=config.evaluation.batch_size)

    def generate(self, dataloader):
        itr = iter(dataloader)
        epoch = 0
        while True:
            try:
                batch = next(itr)
            except StopIteration:
                itr = iter(dataloader)
                batch = next(itr)
                epoch += 1
            batch = shard(batch)
            batch["epoch"] = epoch
            yield batch

    def train_iter(self):
        return self.generate(self.train_dl)

    def validation_iter(self):
        return self.generate(self.val_dl)
