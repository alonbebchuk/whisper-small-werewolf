import numpy as np
import random

from src.data.process_dataset import strategies

strategies_len = len(strategies)


class BertDataCollator:
    def __call__(self, features):
        rand_indices = np.random.randint(0, strategies_len, size=len(features))
        bert_choices = [features[i]["bert_choices"][rand_indices[i]] for i in range(len(features))]

        batch = {
            "strategy": np.array([choice["strategy"] for choice in bert_choices], dtype=object),
            "label": np.array([choice["label"] for choice in bert_choices], dtype=np.int32),
            "input_ids": np.array([choice["input_ids"] for choice in bert_choices], dtype=np.int32),
            "attention_mask": np.array([choice["attention_mask"] for choice in bert_choices], dtype=np.int32),
        }
        return batch


class WhisperDataCollator:
    def __call__(self, features):
        rand_indices = np.random.randint(0, strategies_len, size=len(features))
        whisper_choices = [features[i]["whisper_choices"][rand_indices[i]] for i in range(len(features))]

        batch = {
            "strategy": np.array([choice["strategy"] for choice in whisper_choices], dtype=object),
            "label": np.array([choice["label"] for choice in whisper_choices], dtype=np.int32),
            "input_features": np.array([feature["input_features"] for feature in features], dtype=np.float32),
            "decoder_input_ids": np.array([choice["decoder_input_ids"] for choice in whisper_choices], dtype=np.int32),
            "target_tokens": np.array([choice["target_tokens"] for choice in whisper_choices], dtype=np.int32),
            "attention_mask": np.array([choice["attention_mask"] for choice in whisper_choices], dtype=np.int32),
            "loss_mask": np.array([choice["loss_mask"] for choice in whisper_choices], dtype=np.float32),
        }
        return batch
