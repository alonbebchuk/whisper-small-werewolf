import numpy as np
import os

from datasets import Audio, load_dataset, load_from_disk
from src.new.tokenizers import get_tokenizer
from transformers import WhisperFeatureExtractor

completions = ["No", "Yes"]
prompt_format = "{dialogue}Does the final utterance conform to the strategy: '{strategy}'?\nAnswer with a single word: Yes or No.\n"
strategies = ["Accusation", "Call for Action", "Defense", "Evidence", "Identity Declaration", "Interrogation"]

max_audio_len = 480000
max_dialogue_lookback = 10
max_tokens_len = 448
pad_token_id = -100
sampling_rate = 16000


def get_dialogue_and_targets(sample):
    targets = set(sample["dialogue"][-1]["target"].split(", "))
    dialogue = "".join(f"{d['speaker']}: {d['utterance']}\n" for d in sample["dialogue"][-max_dialogue_lookback:])
    return targets, dialogue


def get_is_strategy_and_prompt(targets, dialogue, strategy):
    is_strategy = int(strategy in targets)
    prompt = prompt_format.format(dialogue=dialogue, strategy=strategy)
    return is_strategy, prompt


def get_bert_process_sample_fn(model_name):
    tokenizer = get_tokenizer(model_name)

    def bert_process_sample_fn(sample):
        targets, dialogue = get_dialogue_and_targets(sample)
        choices = []
        for strategy_id, strategy in enumerate(strategies):
            is_strategy, prompt = get_is_strategy_and_prompt(targets, dialogue, strategy)
            result = tokenizer(prompt, padding="max_length", max_length=max_tokens_len, truncation=True)
            choices.append(
                {
                    "strategy_id": strategy_id,
                    "is_strategy": is_strategy,
                    "labels": is_strategy,
                    "input_ids": result.input_ids,
                    "attention_mask": result.attention_mask,
                }
            )
        return {"choices": choices}

    return bert_process_sample_fn


def get_whisper_process_sample_fn(model_name):
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = get_tokenizer(model_name)

    def whisper_process_sample_fn(sample):
        input_features = feature_extractor(sample["audio"]["array"][-max_audio_len:], sampling_rate=sampling_rate).input_features[0]
        choices = []
        targets, dialogue = get_dialogue_and_targets(sample)
        for strategy_id, strategy in enumerate(strategies):
            is_strategy, prompt = get_is_strategy_and_prompt(targets, dialogue, strategy)
            completion = completions[is_strategy]
            result = tokenizer([prompt + completion], padding="max_length", max_length=max_tokens_len, truncation=True, return_length=True)
            completion_index = result.length[0] - 3
            labels = np.full(max_tokens_len, pad_token_id, dtype=np.int32)
            labels[completion_index] = result.input_ids[0][1:][completion_index]
            choices.append(
                {
                    "strategy_id": strategy_id,
                    "is_strategy": is_strategy,
                    "labels": labels,
                    "decoder_input_ids": result.input_ids[0],
                }
            )
        return {"input_features": input_features, "choices": choices}

    return whisper_process_sample_fn


_dataset = None


def get_dataset(model_name):
    global _dataset
    if _dataset is None:
        if model_name == "bert":
            dataset_path = os.path.join(os.environ["HF_DATASETS_CACHE"], "bert_werewolf_data")
            process_sample_fn = get_bert_process_sample_fn(model_name)
        elif model_name == "whisper":
            dataset_path = os.path.join(os.environ["HF_DATASETS_CACHE"], "whisper_werewolf_data")
            process_sample_fn = get_whisper_process_sample_fn(model_name)
        else:
            raise Exception(f"Model name {model_name} is not supported.")
        if not os.path.exists(dataset_path):
            num_proc = os.cpu_count() // 2
            _dataset = load_dataset("iohadrubin/werewolf_dialogue_data_10sec", num_proc=num_proc)
            _dataset = _dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
            _dataset = _dataset.filter(lambda sample: sample["dialogue"][-1]["target"] is not None, num_proc=num_proc)
            _dataset = _dataset.map(process_sample_fn, num_proc=num_proc)
            _dataset = _dataset.shuffle(seed=42)
            _dataset = _dataset.flatten_indices(num_proc=num_proc)
            _dataset.save_to_disk(dataset_path, num_proc=num_proc)
        _dataset = load_from_disk(dataset_path)
    return _dataset
