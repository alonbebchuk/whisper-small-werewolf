import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


def create_stream(
    model_name="openai/whisper-small",
    bos_token="<|startoftranscript|>",
    dataset="iohadrubin/werewolf_dialogue_data_10sec_v2",
    batch_size=16
):
    tokenizer = WhisperTokenizer.from_pretrained(model_name, bos_token)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    train_werewolf_data = load_dataset(dataset, split="train", streaming=True)

    process_sample_fn = create_process_sample_fn(tokenizer)
    iterable_train_werewolf_data = train_werewolf_data.map(process_sample_fn)
    iterable_train_werewolf_batches = iterable_train_werewolf_data.iter(batch_size)

    collate_fn = create_collate_fn(feature_extractor, processor)
    train_werewolf_batches = (collate_fn(batch) for batch in iterable_train_werewolf_batches)

    return train_werewolf_batches


def create_process_sample_fn(
    tokenizer,
    prompt_prefix_len=65,
    seq_len=448,
    special_token_len=2
):
    def process_sample(sample):
        truncated = False
        prompt_tokens = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        completion_tokens = tokenizer.encode(sample["completion"], add_special_tokens=False)
        completion_len = len(completion_tokens)

        max_prompt_len = seq_len - (completion_len + special_token_len)
        if prompt_len > max_prompt_len:
            truncated = True
            prompt_tokens = prompt_tokens[:prompt_prefix_len] + prompt_tokens[prompt_prefix_len + (prompt_len - max_prompt_len):]
            prompt_len = len(prompt_tokens)

        tokens = [tokenizer.bos_token_id] + prompt_tokens + completion_tokens + [tokenizer.eos_token_id]
        loss_mask = [0.0] * (1 + prompt_len) + [1.0] * (completion_len + 1)

        input_tokens = tokens[:-1]
        input_tokens = input_tokens + [tokenizer.pad_token_id] * (seq_len - len(input_tokens))

        attention_mask = [1.0] * len(input_tokens) + [0.0] * (seq_len - len(input_tokens))

        target_tokens = tokens[1:]
        target_tokens = target_tokens + [tokenizer.pad_token_id] * (seq_len - len(target_tokens))

        loss_mask = loss_mask[1:]
        loss_mask = loss_mask + [0.0] * (seq_len - len(loss_mask))

        result = {
            "input_tokens": np.array(input_tokens, dtype=np.int32),
            "target_tokens": np.array(target_tokens, dtype=np.int32),
            "loss_mask": np.array(loss_mask, dtype=np.float32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
            "truncated": truncated
        }

        return result

    return process_sample


def create_collate_fn(feature_extractor, processor, sampling_rate=16000):
    audio = Audio()

    def collate(batch):
        audio_arrays = [audio.decode_example(x)["array"] for x in batch["audio"]]

        input_features = feature_extractor(audio_arrays, sampling_rate=sampling_rate).input_features
        input_features = processor.feature_extractor.pad([{"input_features": x} for x in list(input_features)], return_tensors="np").input_features

        result = {
            "input_features": input_features,
            "decoder_input_ids": np.array(batch["input_tokens"], dtype=np.int32),
            "target_tokens": np.array(batch["target_tokens"], dtype=np.int32),
            "loss_mask": np.array(batch["loss_mask"], dtype=np.float32),
            "attention_mask": np.array(batch["attention_mask"], dtype=np.int32),
        }

        return result

    return collate
