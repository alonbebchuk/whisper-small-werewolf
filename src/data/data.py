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
        tokens = tokenizer.encode(sample['prompt'] + sample['completion'], add_special_tokens=False)
        truncated = False
        if len(tokens) > seq_len:
            assert False, "Holy shit, what is this?, {}".format(len(tokens))
            tokens = tokens[:seq_len]
            truncated = True
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        prompt_len = len(tokenizer.encode(sample['prompt'], add_special_tokens=False)) + 1  # add bos token
        loss_masks = ([0.0] * prompt_len) + ([1.0] * (len(tokens) - prompt_len))
        # trunacte and pad everything out
        assert len(tokens)<=  seq_len+1, "WTF"
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
            loss_masks = loss_masks[:seq_len]
        # before padding, account for shifting
        input_tokens = tokens[:-1]
        loss_masks = loss_masks[1:]
        target_tokens = tokens[1:]
        attention_mask = [1] * len(input_tokens) + [0] * (seq_len - len(input_tokens))
        input_tokens = input_tokens + [tokenizer.pad_token_id] * (seq_len - len(input_tokens))
        target_tokens = target_tokens + [tokenizer.pad_token_id] * (seq_len - len(target_tokens))
        loss_masks = loss_masks + [0.0] * (seq_len - len(loss_masks))
        return {
            "input_tokens": np.array(input_tokens, dtype=np.int32),
            "target_tokens": np.array(target_tokens, dtype=np.int32),
            "loss_mask": np.array(loss_masks, dtype=np.float32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
            "truncated": truncated,
        }


    return process_sample


def create_collate_fn(feature_extractor, processor, sampling_rate=16000):
    audio = Audio()

    def collate(batch):
        audio_arrays = []
        for x in batch["audio"]:
            if "array" in x:
                audio_arrays.append(x["array"])
            else:
                audio_arrays.append(audio.decode_example(x)["array"])

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
