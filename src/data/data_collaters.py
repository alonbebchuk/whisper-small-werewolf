import flax
import numpy as np
import random

from transformers import BertTokenizer, WhisperTokenizer

max_len = 448

strategies = ["Accusation", "Call for Action", "Defense", "Evidence", "Identity Declaration", "Interrogation"]

bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", bos_token="<|startoftranscript|>")


@flax.struct.dataclass
class BertDataCollator:
    def __call__(self, features):
        strategy_list, label_list, prompt_list = [], [], []
        for feature in features:
            strategy = random.choice(strategies)
            label = int(strategy in feature["target_strategies"])
            prompt = bert_tokenizer.decode(bert_tokenizer.encode(f"{feature['dialogue']}{strategy}: ", add_special_tokens=False)[-(max_len - 2) :])
            strategy_list.append(strategy)
            label_list.append(label)
            prompt_list.append(prompt)

        batch = bert_tokenizer(prompt_list, padding="max_length", max_length=max_len, return_tensors="np")

        return {
            "strategy": np.array(strategy_list, dtype=object),
            "labels": np.array(label_list, dtype=np.int32),
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
        }


@flax.struct.dataclass
class WhisperDataCollator:
    def __call__(self, features):
        strategy_list, completion_list, prompt_list = [], [], []
        for feature in features:
            strategy = random.choice(strategies)
            completion = "Yes" if strategy in feature["target_strategies"] else "No"
            prompt = f"{feature['dialogue']}{strategy}: "
            strategy_list.append(strategy)
            completion_list.append(completion)
            prompt_list.append(prompt)

        completion_tokens_batch = whisper_tokenizer(completion_list, add_special_tokens=False)["input_ids"]
        prompt_tokens_batch = whisper_tokenizer(prompt_list, add_special_tokens=False)["input_ids"]

        decoder_input_ids_list = []
        target_tokens_list = []
        attention_mask_list = []
        loss_mask_list = []
        for i in range(len(features)):
            completion_tokens = completion_tokens_batch[i]
            completion_tokens_len = len(completion_tokens)
            prompt_tokens = prompt_tokens_batch[i][-(max_len - completion_tokens_len - 1) :]
            prompt_tokens_len = len(prompt_tokens)

            tokens = prompt_tokens + completion_tokens
            non_padding_len = len(tokens) + 1
            padding_len = max_len - non_padding_len
            decoder_input_ids = [whisper_tokenizer.bos_token_id] + tokens + ([whisper_tokenizer.pad_token_id] * padding_len)
            target_tokens = tokens + [whisper_tokenizer.eos_token_id] + ([whisper_tokenizer.pad_token_id] * padding_len)
            attention_mask = ([1] * non_padding_len) + ([0] * padding_len)
            loss_mask = ([0.0] * prompt_tokens_len) + ([1.0] * completion_tokens_len) + [1.0] + ([0.0] * padding_len)

            decoder_input_ids_list.append(decoder_input_ids)
            target_tokens_list.append(target_tokens)
            attention_mask_list.append(attention_mask)
            loss_mask_list.append(loss_mask)

        return {
            "strategy": np.array(strategy_list, dtype=object),
            "completion": np.array(completion_list, dtype=object),
            "input_features": np.array([feature["input_features"] for feature in features], dtype=np.float32),
            "decoder_input_ids": np.array(decoder_input_ids_list, dtype=np.int32),
            "target_tokens": np.array(target_tokens_list, dtype=np.int32),
            "attention_mask": np.array(attention_mask_list, dtype=np.int32),
            "loss_mask": np.array(loss_mask_list, dtype=np.float32),
        }
