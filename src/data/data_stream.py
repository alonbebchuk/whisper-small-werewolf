import jax
import numpy as np

from datasets import Audio, ClassLabel, concatenate_datasets, load_dataset
from flax.training.common_utils import shard
from torch.utils.data import DataLoader


def create_get_target_strategies_fn(config):
    def get_target_strategies(sample):
        target = sample["dialogue"][-1]["target"]
        if target is None:
            target_strategies = None
        else:
            target_strategies = set([target.strip() for target in sample["dialogue"][-1]["target"].split(", ")])
            target_strategies = target_strategies & config.dataset.strategies
        return {"target_strategies": target_strategies}

    return get_target_strategies


def get_prompts(config, tokenizer):
    prompts = {}
    for strategy in config.dataset.strategies:
        prompt_prefix = config.dataset.prompt_prefix_format.format(strategy=strategy)
        prompt_suffix = config.dataset.prompt_suffix_format.format(strategy=strategy)
        prompt_format = prompt_prefix + "{dialogue}" + prompt_suffix if config.dataset.use_dialogue else prompt_prefix + prompt_suffix
        prompt_len = len(tokenizer.encode(prompt_prefix + prompt_suffix, add_special_tokens=False))
        prompts[strategy] = {"format": prompt_format, "length": prompt_len}
    return prompts


def create_get_prompt_and_completion_fn(config, prompts, tokenizer):
    def get_prompt_and_completion(sample):
        strategy = sample["strategy"]
        completion = "Yes" if strategy in sample["target_strategies"] else "No"
        if config.dataset.use_dialogue:
            remaining_len = config.model.max_seq_len - prompts[strategy]["length"]
            dialogues = []
            for d in reversed(sample["dialogue"]):
                dialogue = f"{d['speaker']}: {d['utterance']}\n"
                dialogue_len = len(tokenizer.encode(dialogue, add_special_tokens=False))
                if dialogue_len >= remaining_len:
                    break
                dialogues.append(dialogue)
                remaining_len -= dialogue_len
            prompt = None if len(dialogues) == 0 else prompts[strategy]["format"].format(dialogue="".join(reversed(dialogues)))
            prompt_len = config.model.max_seq_len - remaining_len
        else:
            prompt = prompts[strategy]["format"]
            prompt_len = prompts[strategy]["length"]
        return {"prompt": prompt, "prompt_len": prompt_len, "completion": completion}

    return get_prompt_and_completion


class DataStream:
    def __init__(self, config, tokenizer, collate_fn):
        super().__init__()
        self.dataset = load_dataset(config.dataset.name, streaming=True)
        target_strategies_fn = create_get_target_strategies_fn(config)
        process_count, process_index = jax.process_count(), jax.process_index()
        prompts = get_prompts(config, tokenizer)
        get_prompt_and_completion_fn = create_get_prompt_and_completion_fn(config, prompts, tokenizer)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(target_strategies_fn)
            self.dataset[split] = self.dataset[split].filter(lambda sample: sample["target_strategies"] is not None)
            self.dataset[split] = concatenate_datasets([self.dataset[split].map(lambda _: {"strategy": strategy}) for strategy in config.dataset.strategies])
            self.dataset[split] = self.dataset[split].shuffle()
            self.dataset[split] = self.dataset[split].shard(num_shards=process_count, index=process_index)
            self.dataset[split] = self.dataset[split].map(get_prompt_and_completion_fn)
            batch_size = config.training.batch_size if split == "train" else config.evaluation.batch_size
            self.dataset[split] = DataLoader(self.dataset[split], collate_fn=collate_fn, batch_size=batch_size, num_workers=10, prefetch_factor=5, drop_last=True)

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
            non_array_elements = {key: batch.pop(key) for key, value in list(batch.items()) if not isinstance(value, np.ndarray)}
            batch = shard(batch)
            batch["epoch"] = epoch
            for key, value in non_array_elements.items():
                batch[key] = value
            yield batch

    def get_iter(self, split):
        return self.generate(self.dataset[split])


def create_bert_collate_fn(config, tokenizer):
    ClassLabels = ClassLabel(num_classes=2, names=["No", "Yes"])

    def bert_collate_fn(batch):
        results ={}
        prompts = [sample["prompt"] for sample in batch]
        # Assert that prompts is a list of strings
        print(prompts)
        assert isinstance(prompts, list), f"Expected prompts to be a list, but got {type(prompts)}"
        assert all(isinstance(prompt, str) for prompt in prompts), "All elements in prompts must be strings"
        out = tokenizer.batch_encode_plus(prompts, padding="max_length", max_length=config.model.max_seq_len, return_tensors="np", truncation=True)
        # Assert that input_ids has the expected shape
        input_ids = out.input_ids
        attention_mask = out.attention_mask
        # print(input_ids.shape)

        assert input_ids.shape == (len(batch), config.model.max_seq_len), f"Expected shape {(len(batch), config.model.max_seq_len)} but got {input_ids.shape}"
        results["input_ids"] = input_ids
        results["attention_mask"] = attention_mask
        results["labels"] = np.array(ClassLabels.str2int([sample["completion"] for sample in batch]), dtype=np.int32)
        results["strategy"] = [sample["strategy"] for sample in batch]
        return results

    return bert_collate_fn


class BertDataStream(DataStream):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer, create_bert_collate_fn(config, tokenizer))


def create_whisper_collate_fn(config, tokenizer, feature_extractor):
    audio = Audio()

    def whisper_collate_fn(batch):
        input_features, decoder_input_ids, target_tokens, attention_mask, loss_mask, strategy = [], [], [], [], [], []
        for sample in batch:
            if config.dataset.use_audio:
                audio_array = sample["audio"]["array"] if "array" in sample["audio"] else audio.decode_example(sample["audio"])["array"]
                if len(audio_array) > config.model.max_audio_array_len:
                    audio_array = audio_array[-config.model.max_audio_array_len :]
            else:
                audio_array = np.zeros(config.model.max_audio_array_len, np.float32)
            input_features.append(feature_extractor(audio_array, sampling_rate=config.model.sampling_rate).input_features[0])
            
            tokens = [tokenizer.bos_token_id] + tokenizer.encode(sample["prompt"] + sample["completion"], add_special_tokens=False) + [tokenizer.eos_token_id]
            slice_len = len(tokens) - 1
            padding_len = config.model.max_seq_len - slice_len
            
            prompt_len = sample["prompt_len"]
            decoder_input_ids.append(tokens[:-1] + [tokenizer.pad_token_id] * padding_len)
            target_tokens.append(tokens[1:] + [tokenizer.pad_token_id] * padding_len)
            attention_mask.append([1] * slice_len + [0] * padding_len)
            loss_mask.append([0.0] * prompt_len + [1.0] * (slice_len - prompt_len) + [0.0] * padding_len)
            strategy.append(sample["strategy"])
        return {"input_features": np.array(input_features, dtype=np.float32), 
                "decoder_input_ids": np.array(decoder_input_ids, dtype=np.int32), 
                "target_tokens": np.array(target_tokens, dtype=np.int32), 
                "loss_mask": np.array(loss_mask, dtype=np.float32), 
                "attention_mask": np.array(attention_mask, dtype=np.int32), 
                "strategy": strategy}

    return whisper_collate_fn


class WhisperDataStream(DataStream):
    def __init__(self, config, tokenizer, feature_extractor):
        super().__init__(config, tokenizer, create_whisper_collate_fn(config, tokenizer, feature_extractor))
