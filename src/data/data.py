import numpy as np
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


def create_dataloader(model_name="openai/whisper-small", bos_token="<|startoftranscript|>", dataset="iohadrubin/werewolf_dialogue_data_10sec", batch_size=16):
    tokenizer = WhisperTokenizer.from_pretrained(model_name, bos_token=bos_token)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    train_werewolf_data = load_dataset(dataset, split="train")

    into_prompt_completion_fn = create_into_prompt_completion_fn(tokenizer)
    process_sample_fn = create_process_sample_fn(tokenizer)
    train_werewolf_data = filter_data(train_werewolf_data)
    train_werewolf_data = train_werewolf_data.map(into_prompt_completion_fn, batched=True, batch_size=1)
    train_werewolf_data = train_werewolf_data.filter(lambda x: x["prompt"] is not None)
    train_werewolf_data = train_werewolf_data.map(process_sample_fn)

    shuffled_iterable_dataset = train_werewolf_data.shuffle()

    train_werewolf_dataloader = DataLoader(shuffled_iterable_dataset, collate_fn=create_collate_fn(feature_extractor, processor), batch_size=batch_size)

    return train_werewolf_dataloader


def filter_data(werewolf_data, max_duration=30):
    def filter_fn(x):
        duration = x["end"] - x["start"]
        if duration > max_duration:
            return False
        target = x["dialogue"][-1]["target"]
        if target is None or len(target.strip()) == 0:
            return False
        return True

    werewolf_data = werewolf_data.filter(filter_fn)
    return werewolf_data


strategies = ["Accusation", "Defense", "Evidence", "Identity Declaration", "Interrogation", "No Strategy", "Call for Action"]
prompt_format = """Given the previous audio and the following dialogue, determine whether the last utterance in the following spoken dialogue fits under the strategy category of: {strategy}.
Respond with a single word: Yes or No.
Dialogue:
```
{dialogue}
```
Does the last utterance fit the strategy category {strategy}?
Completion:
"""


def create_into_prompt_completion_fn(tokenizer, seq_len=448):
    max_length = seq_len - 1

    def into_prompt_completion(sample):
        sample = sample[0]
        i = 0
        prompts, strategies, completions = [], [], []
        targets = sample["dialogue"][-1]["target"].split(", ")
        # TODO: move to v2 and take dialogue from there
        for strategy in strategies:
            completion = "Yes" if strategy in targets else "No"
            prompt = None
            while i < len(sample["dialogue"]):
                curr_dialogue = sample["dialogue"][i:]
                dialogue = "\n".join(f"{x['speaker']}: {x['utterance']}" for x in curr_dialogue)
                prompt = prompt_format.format(strategy=strategy, dialogue=dialogue)
                input_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
                if len(input_ids) <= max_length:
                    break
                i += 1
            prompts.append(prompt)
            strategies.append(strategy)
            completions.append(completion)
        return {"prompt": prompts, "strategy": strategies, "completion": completions}

    return into_prompt_completion


def create_process_sample_fn(tokenizer, seq_len=448):
    def process_sample(sample):
        tokens = tokenizer.encode(sample["prompt"] + sample["completion"], add_special_tokens=False)
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        prompt_len = len(tokenizer.encode(sample["prompt"], add_special_tokens=False)) + 1
        loss_masks = ([0.0] * prompt_len) + ([1.0] * (len(tokens) - prompt_len))
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
