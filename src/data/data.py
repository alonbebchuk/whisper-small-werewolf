import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from torch.utils.data import DataLoader


def create_dataloader(model_name="openai/whisper-small", bos_token="<|startoftranscript|>", dataset="alonbeb/werewolf_Defense_data", batch_size=16):
    tokenizer = WhisperTokenizer.from_pretrained(model_name, bos_token=bos_token)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    train_werewolf_data = load_dataset(dataset, split="train", streaming=True)

    process_sample_fn = create_process_sample_fn(tokenizer)
    train_werewolf_data = train_werewolf_data.map(process_sample_fn)
    train_werewolf_batches = train_werewolf_data.iter(batch_size)

    collate_fn = create_collate_fn(feature_extractor, processor)
    train_werewolf_dataloader = DataLoader(train_werewolf_batches, collate_fn=collate_fn, batch_size=batch_size)

    return train_werewolf_dataloader


def create_process_sample_fn(tokenizer, seq_length=448):
    def process_sample(sample):
        tokens = tokenizer.encode(sample['prompt'] + sample['completion'], add_special_tokens=False)
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

        input_tokens = tokens[:-1]
        input_tokens_len = len(input_tokens)
        input_tokens = input_tokens + [tokenizer.pad_token_id] * (seq_length - len(input_tokens))

        attention_mask = [1] * input_tokens_len + [0] * (seq_length - input_tokens_len)

        target_tokens = tokens[1:]
        target_tokens = target_tokens + [tokenizer.pad_token_id] * (seq_length - len(target_tokens))

        prompt_len = len(tokenizer.encode(sample['prompt'], add_special_tokens=False)) + 1
        loss_mask = ([0.0] * prompt_len) + ([1.0] * (len(tokens) - prompt_len))
        loss_mask = loss_mask[1:]
        loss_mask = loss_mask + [0.0] * (seq_length - len(loss_mask))

        result ={
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
