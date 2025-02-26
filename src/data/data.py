import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


def create_stream(model_name="openai/whisper-small", bos_token="<|startoftranscript|>", dataset="alonbeb/werewolf_dataset", batch_size=16):
    tokenizer = WhisperTokenizer.from_pretrained(model_name, bos_token=bos_token)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    train_werewolf_data = load_dataset(dataset, split="train", streaming=True)

    process_sample_fn = create_process_sample_fn(tokenizer)
    iterable_train_werewolf_data = train_werewolf_data.map(process_sample_fn)
    iterable_train_werewolf_batches = iterable_train_werewolf_data.iter(batch_size)

    collate_fn = create_collate_fn(feature_extractor, processor)
    train_werewolf_batches = (collate_fn(batch) for batch in iterable_train_werewolf_batches)

    return train_werewolf_batches


def create_process_sample_fn(tokenizer, seq_len=448):
    def process_sample(sample):
        prompt_tokens = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        completion_tokens = tokenizer.encode(sample["completion"], add_special_tokens=False)

        input_tokens = [tokenizer.bos_token_id] + prompt_tokens + completion_tokens
        input_tokens_padding_len = seq_len - len(input_tokens)

        input_tokens = input_tokens + ([tokenizer.pad_token_id] * input_tokens_padding_len)
        attention_mask = ([1.0] * (seq_len - input_tokens_padding_len)) + ([0.0] * input_tokens_padding_len)

        target_tokens = prompt_tokens + completion_tokens
        target_tokens_padding_len = seq_len - len(target_tokens) - 1

        target_tokens = target_tokens + ([tokenizer.pad_token_id] * target_tokens_padding_len) + [tokenizer.eos_token_id]

        completion_tokens_len = len(completion_tokens) + 1
        loss_mask = ([0.0] * (seq_len - completion_tokens_len)) + ([1.0] * (len(completion_tokens) + 1))

        loss_mask = loss_mask[1:]
        loss_mask_padding_len = seq_len - len(loss_mask)

        loss_mask = loss_mask + [0.0] * loss_mask_padding_len

        result = {
            "input_tokens": np.array(input_tokens, dtype=np.int32),
            "target_tokens": np.array(target_tokens, dtype=np.int32),
            "loss_mask": np.array(loss_mask, dtype=np.float32),
            "attention_mask": np.array(attention_mask, dtype=np.int32)
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
