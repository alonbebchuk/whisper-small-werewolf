import numpy as np
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


def create_dataloader(model_name="openai/whisper-small", bos_token="<|startoftranscript|>", dataset="iohadrubin/werewolf_dialogue_data_10sec_v2", batch_size=16):
    tokenizer = WhisperTokenizer.from_pretrained(model_name, bos_token=bos_token)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    train_werewolf_data = load_dataset(dataset, split="train")

    process_sample_fn = create_process_batch_fn(tokenizer)
    train_werewolf_data = train_werewolf_data.map(process_sample_fn, batched=True, batch_size=batch_size, remove_columns=train_werewolf_data.column_names)

    shuffled_iterable_dataset = train_werewolf_data.shuffle()

    train_werewolf_dataloader = DataLoader(shuffled_iterable_dataset, collate_fn=create_collate_fn(feature_extractor, processor), batch_size=batch_size)

    return train_werewolf_dataloader


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


def create_process_batch_fn(tokenizer, seq_len=448):
    def process_batch(batch):
        audio_list, prompt_list, strategy_list, completion_list, input_tokens_list, target_tokens_list, loss_mask_list, attention_mask_list = [], [], [], [], [], [], [], []

        audio = [a for a in batch["audio"]]
        dialogue = [p.split("```")[1].strip() for p in batch["prompt"]]
        targets = [c.split(", ") for c in batch["completion"]]
        for strategy in strategies:
            prompt = [prompt_format.format(strategy=strategy, dialogue=d) for d in dialogue]
            completion = ["Yes" if strategy in t else "No" for t in targets]
            text = [p + c for (p, c) in zip(prompt, completion)]

            tokens = [[tokenizer.bos_token_id] + tokenizer.encode(t, add_special_tokens=False) + [tokenizer.eos_token_id] for t in text]

            input_tokens = [t[:-1] for t in tokens]
            input_tokens = [it + [tokenizer.pad_token_id] * (seq_len - len(it)) for it in input_tokens]
            input_tokens = [np.array(it, dtype=np.int32) for it in input_tokens]

            target_tokens = [t[1:] for t in tokens]
            target_tokens = [tt + [tokenizer.pad_token_id] * (seq_len - len(tt)) for tt in target_tokens]
            target_tokens = [np.array(tt, dtype=np.int32) for tt in target_tokens]

            tokens_len = [len(t) for t in tokens]
            prompt_len = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompt]
            loss_mask = [([0.0] * p_len) + ([1.0] * (t_len - p_len)) + ([0.0] * (seq_len - t_len)) for p_len, t_len in zip(prompt_len, tokens_len)]
            loss_mask = [np.array(lm, dtype=np.float32) for lm in loss_mask]

            input_tokens_len = [len(it) for it in input_tokens]
            attention_mask = [([1] * it_len + [0] * (seq_len - it_len)) for it_len in input_tokens_len]
            attention_mask = [np.array(am, dtype=np.int32) for am in attention_mask]

            audio_list.extend(audio)
            prompt_list.extend(prompt)
            strategy_list.extend([strategy] * 16)
            completion_list.extend(completion)
            input_tokens_list.extend(input_tokens)
            target_tokens_list.extend(target_tokens)
            loss_mask_list.extend(loss_mask)
            attention_mask_list.extend(attention_mask)

        return {"audio": audio_list, "prompt": prompt_list, "strategy": strategy_list, "completion": completion_list, "input_tokens": input_tokens_list, "target_tokens": target_tokens_list, "loss_mask": loss_mask_list, "attention_mask": attention_mask_list}

    return process_batch


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
