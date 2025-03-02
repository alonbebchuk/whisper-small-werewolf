import jax
import numpy as np

from datasets import load_dataset, interleave_datasets, DatasetDict, Audio
from flax.training.common_utils import shard
from src.common.config import get_strategy_dataset_name, strategies, prompt_without_dialogue_format
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


def create_process_sample_fn(config, use_dialogue):
    tokenizer = WhisperTokenizer.from_pretrained(config.model.name, bos_token=config.model.bos_token)

    def process_sample(sample):
        prompt = sample["prompt"] if use_dialogue else prompt_without_dialogue_format.format(sample["strategy"])
        tokens = tokenizer.encode(prompt + sample["completion"], add_special_tokens=False)
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

        slice_len = len(tokens) - 1
        padding_len = config.model.max_len - slice_len
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        input_tokens = tokens[:-1] + [tokenizer.pad_token_id] * padding_len
        target_tokens = tokens[1:] + [tokenizer.pad_token_id] * padding_len
        attention_mask = [1] * slice_len + [0] * padding_len
        loss_mask = [0.0] * prompt_len + [1.0] * (slice_len - prompt_len) + [0.0] * padding_len

        input_tokens = np.array(input_tokens, dtype=np.int32)
        target_tokens = np.array(target_tokens, dtype=np.int32)
        loss_mask = np.array(loss_mask, dtype=np.float32)
        attention_mask = np.array(attention_mask, dtype=np.int32)

        result = {"input_tokens": input_tokens, "target_tokens": target_tokens, "loss_mask": loss_mask, "attention_mask": attention_mask}
        return result

    return process_sample


def create_collate_fn(config, use_audio):
    audio = Audio()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model.name)
    processor = WhisperProcessor.from_pretrained(config.model.name)

    def collate(batch):
        audio_arrays = []
        for x in batch:
            if "array" in x["audio"]:
                audio_arrays.append(x["audio"]["array"])
            else:
                audio_arrays.append(audio.decode_example(x["audio"])["array"])

        input_features = feature_extractor(audio_arrays, sampling_rate=config.model.sampling_rate).input_features
        input_features = processor.feature_extractor.pad([{"input_features": x} for x in list(input_features)], return_tensors="np").input_features
        if not use_audio:
            input_features = np.zeros(input_features, np.float32)
        decoder_input_ids = np.array([x["input_tokens"] for x in batch], dtype=np.int32)
        target_tokens = np.array([x["target_tokens"] for x in batch], dtype=np.int32)
        loss_mask = np.array([x["loss_mask"] for x in batch], dtype=np.float32)
        attention_mask = np.array([x["attention_mask"] for x in batch], dtype=np.int32)

        result = {"input_features": input_features, "decoder_input_ids": decoder_input_ids, "target_tokens": target_tokens, "loss_mask": loss_mask, "attention_mask": attention_mask}
        return result

    return collate


class DataStream:
    def __init__(self, config, use_audio, use_dialogue):
        super().__init__()

        train_ds = interleave_datasets([load_dataset(get_strategy_dataset_name(config, strategy), split="train", streaming=True) for strategy in strategies])
        val_ds = interleave_datasets([load_dataset(get_strategy_dataset_name(config, strategy), split="validation", streaming=True) for strategy in strategies])

        process_count, process_index = jax.process_count(), jax.process_index()
        train_ds = train_ds.shard(num_shards=process_count, index=process_index)
        val_ds = val_ds.shard(num_shards=process_count, index=process_index)

        process_sample_fn = create_process_sample_fn(config, use_dialogue)
        train_ds = train_ds.map(process_sample_fn)
        val_ds = val_ds.map(process_sample_fn)

        collate_fn = create_collate_fn(config, use_audio)
        self.train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=config.training.batch_size, num_workers=10, prefetch_factor=5, drop_last=True)
        self.val_dl = DataLoader(val_ds, collate_fn=collate_fn, batch_size=config.evaluation.batch_size, num_workers=10, prefetch_factor=5, drop_last=True)

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
