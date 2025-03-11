import numpy as np

from src.new.datasets import strategies

strategies_len = len(strategies)


def get_random_choices(features):
    rand_indices = np.random.randint(0, strategies_len, size=len(features))
    return [features[i]["choices"][rand_indices[i]] for i in range(len(features))]


class BertDataCollator:
    def __call__(self, features):
        bert_choices = get_random_choices(features)
        return {
            "strategy_id": np.array([choice["strategy_id"] for choice in bert_choices], dtype=np.int32),
            "is_strategy": np.array([choice["is_strategy"] for choice in bert_choices], dtype=np.int32),
            "labels": np.array([choice["labels"] for choice in bert_choices], dtype=np.int32),
            "input_ids": np.array([choice["input_ids"] for choice in bert_choices], dtype=np.int32),
            "attention_mask": np.array([choice["attention_mask"] for choice in bert_choices], dtype=np.int32),
        }


class WhisperDataCollator:
    def __call__(self, features):
        whisper_choices = get_random_choices(features)
        return {
            "input_features": np.array([feature["input_features"] for feature in features], dtype=np.float32),
            "strategy_id": np.array([choice["strategy_id"] for choice in whisper_choices], dtype=np.int32),
            "is_strategy": np.array([choice["is_strategy"] for choice in whisper_choices], dtype=np.int32),
            "labels": np.array([choice["labels"] for choice in whisper_choices], dtype=np.int32),
            "decoder_input_ids": np.array([choice["decoder_input_ids"] for choice in whisper_choices], dtype=np.int32),
        }


_data_collator = None


def get_data_collator(model_name):
    global _data_collator
    if _data_collator is None:
        if model_name == "bert":
            _data_collator = BertDataCollator()
        elif model_name == "whisper":
            _data_collator = WhisperDataCollator()
        else:
            raise Exception(f"Model name {model_name} is not supported.")
    return _data_collator
